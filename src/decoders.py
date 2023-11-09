from typing import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channel: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channel, seq_len)
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channel: int | None = None,
        norm: Callable = nn.BatchNorm1d,
        se: bool = False,
        res: bool = False,
    ) -> None:
        super().__init__()
        self.res = res
        if mid_channel is None:
            mid_channel = out_channels

        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channel, kernel_size=3, padding=1, bias=False),
            norm(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            non_linearity,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        norm: Callable = nn.BatchNorm1d,
        se: bool = False,
        res: bool = False,
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        scale_factor: int = 2,
        norm: Callable = nn.BatchNorm1d,
    ) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, norm=norm
            )
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diff // 2, diff - diff // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def create_layer_norm(channel: int, length: int) -> nn.LayerNorm:
    return nn.LayerNorm([channel, length])


class Unet1DDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        duration: int,
        bilinear: bool = True,
        se: bool = False,
        res: bool = False,
        scale_factor: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.duration = duration
        self.bilinear = bilinear
        self.se = se
        self.res = res
        self.scale_factor = scale_factor

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(
            self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration)
        )
        self.down1 = Down(
            64,
            128,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )

        self.down2 = Down(
            128,
            256,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )

        self.down3 = Down(
            256,
            512,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )

        self.down4 = Down(
            512,
            1024 // factor,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 16),
        )

        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )

        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )

        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )

        self.up4 = Up(
            128,
            64,
            bilinear,
            self.scale_factor,
            norm=partial(create_layer_norm, length=self.duration),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )

        # self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward

        Args:
            x: (batch_size, n_channels, n_timesteps)
            labels: (batch_size, n_classes, n_timestamps)

        Returns:
            (batch_size, n_classes, n_timestamps)
        """

        # 1D Unet
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # (batch_size, n_classes, n_timestamps)
        x = self.cls(x)
        return x.transpose(1, 2)
