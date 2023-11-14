from typing import Sequence

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import initialization as smp_binit
from segmentation_models_pytorch.base import modules as smp_bmd
from segmentation_models_pytorch.base import heads as smp_bheads
from logging import getLogger


logger = getLogger(__name__)


class DecoderBlock(nn.Module):
    """U-Net decoder from Segmentation Models PyTorch

    Ref:
    - https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        conv_in_channels = in_channels + skip_channels
        self.conv1 = smp_bmd.Conv2dReLU(
            in_channels=conv_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = smp_bmd.Conv2dReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.dropout_skip = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        # upsample 2x
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            skipped = self.dropout_skip(skip)
            x = torch.cat([x, skipped], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int],
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # encoder: [64, 96, 192, 384, 768]
        # decoder: [256, 128, 64, 32, 16]

        # [768, 384, 192, 96, 64] -> [64, 96, 192, 384, 768]
        encoder_channels = encoder_channels[::-1]
        # -- computing blocks input and output channels
        # [64]
        head_channels = encoder_channels[0]
        # [64, 256, 128, 64, 32]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        # [96, 192, 384, 768, 0]
        skip_channels = list(encoder_channels[1:]) + [0]
        # [256, 128, 64, 32, 16]
        out_channels = decoder_channels
        self.center = nn.Identity()

        # -- Combine decoder keyword arguments
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
                for (in_ch, skip_ch, out_ch) in zip(
                    in_channels, skip_channels, out_channels
                )
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of tensors from encoder.
                    (256x256, 128x128, 64x64, 32x32, 16x16)
        """
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        # 0 torch.Size([8, 256, 32, 32])
        # 1 torch.Size([8, 128, 64, 64])
        # 2 torch.Size([8, 64, 128, 128])
        # 3 torch.Size([8, 32, 256, 256])
        # 4 torch.Size([8, 16, 512, 512])
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


def _check_reduction(reduction_factors: Sequence[int]) -> None:
    r_prev = 1
    for r in reduction_factors:
        if r / r_prev != 2:
            raise ValueError(
                "Reduction factor of each block should be divisible by the previous one"
            )
        r_prev = r


def _init_timm_encoder(name: str, pretrained: bool) -> nn.Module:
    if name.startswith("coat"):
        return timm.create_model(name, return_interm_layers=True, pretrained=pretrained)
    return timm.create_model(name, features_only=True, pretrained=pretrained)


class CustomUnet(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        decoder_channels: list[int] = [256, 128, 64, 32, 16],
        n_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = _init_timm_encoder(name, pretrained)

        encoder_channels = self.encoder.feature_info.channels()  # type: ignore
        if len(encoder_channels) != len(decoder_channels):
            raise ValueError(
                "Encoder channels and decoder channels should have the same length"
            )
        _check_reduction(self.encoder.feature_info.reduction())  # type: ignore
        logger.info(name, encoder_channels, decoder_channels)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
            use_batchnorm=True,
        )
        self.segmentation_head = smp_bheads.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )
        smp_binit.initialize_decoder(self.decoder)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        logits = self.segmentation_head(decoder_output)

        out = {
            "logits": logits,
            "cls_logits": None,
        }
        return out


def _test_model():
    # m = CustomUnet("efficientnet_b0", pretrained=False)
    m = CustomUnet("maxvit_tiny_rw_256", pretrained=False)
    # m = CustomUnet("maxvit_tiny_rw_224", pretrained=False)
    # m = CustomUnet("maxvit_tiny_tf_384", pretrained=False)

    # x = torch.randn(2, 3, 256, 256)
    # x = torch.randn(2, 3, 64 * 4, 24 * 60 * 4 + 128)
    x = torch.randn(2, 3, 16 * 4 * 4, 32 * 16 * 12)  # w % 32 == 0 and w % 16 == 0
    print(x.shape)
    out = m(x)
    print(out["logits"].shape)


if __name__ == "__main__":
    _test_model()
