from typing import Callable

import torch
import torch.nn as nn
import torchaudio.transforms as TAT


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram.

        Args:
            x (torch.Tensor): (batch_size, in_channels, height=freq, width)

        Returns:
            torch.Tensor: (batch_size, in_channels, height, width)

        References:
        [1]
        https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states/blob/main/src/models/feature_extractor/spectrogram.py
        """
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x - min_) / (max_ - min_ + self.eps)


class SpecFeatureExtractor(nn.Module):
    """Spectrogram feature extractor.

    References:
    [1]
    https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states/blob/main/src/models/feature_extractor/spectrogram.py
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        hop_length: int,
        win_length: int | None = None,
        out_size: int | None = None,
    ) -> None:
        super().__init__()
        self.height = height
        self.out_chans = in_channels
        n_fft = (2 * height) - 1
        self.feature_extractor = nn.Sequential(
            TAT.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length),
            TAT.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels - 1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = self.feature_extractor(x)
        img = self.conv2d(img)
        if self.out_size is not None:
            img = self.pool(img)
        return img


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


def _test_spec_feature_extractor() -> None:
    batch_size = 2
    in_channels = 4
    height = 64
    win_length = 32
    hop_length = 256
    # out_size = 64 * 2
    out_size = 24 * 60 * 4 // 2

    num_features = 4
    seq_len = 24 * 60 * 4

    x = torch.randn(batch_size, num_features, seq_len)
    model = SpecFeatureExtractor(
        in_channels=in_channels,
        height=height,
        hop_length=hop_length,
        win_length=win_length,
        out_size=out_size,
    )
    out = model(x)
    print(f"{out.shape=}")
    assert out.shape == (
        batch_size,
        num_features,
        height,
        out_size,
    ), f"{out.shape=} != {(batch_size, num_features, height, out_size)}"


if __name__ == "__main__":
    _test_spec_feature_extractor()
