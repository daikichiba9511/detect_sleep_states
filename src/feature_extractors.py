from typing import Callable
import torch
import torch.nn as nn


class CNNSpectgram(nn.Module):
    """

    Refs:
    [1]
    https://github.com/analokmaus/kaggle-g2net-public/blob/main/models1d_pytorch/wavegram.py
    [2]
    https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states/blob/main/src/models/feature_extractor/cnn.py

    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int | tuple[int, ...] = 128,
        kernel_size: tuple[int, ...] = (32, 16, 4, 2),
        stride: int = 4,
        sigmoid: bool = False,
        output_size: int | None = None,
        conv: Callable = nn.Conv1d,
        reinit: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_chans = len(kernel_size)
        self.out_size = output_size
        self.sigmoid = sigmoid

        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])

        self.height = base_filters[-1]
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_block = [
                conv(
                    in_channels=in_channels,
                    out_channels=base_filters[0],
                    kernel_size=kernel_size[i],
                    stride=stride,
                    padding=(kernel_size[i] - 1) // 2,
                )
            ]

            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_size[i],
                            stride=stride,
                            padding=(kernel_size[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward to make a spectrogram image.

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """

        # x: (batch_size, in_channels, time_steps)
        out: list[torch.Tensor] = []
        for i in range(self.out_chans):
            # (batch_size, base_filters[i], time_steps // stride)
            spec_out = self.spec_conv[i](x)
            out.append(spec_out)

        # img: (batch_size, out_chans, height=base_filters, time_steps)
        img = torch.stack(out, dim=1)
        if self.out_size is not None:
            img = self.pool(img)  # (batch_size, out_chans, height, out_size)
        if self.sigmoid:
            img = torch.sigmoid(img)
        return img
