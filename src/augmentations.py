from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as TAT


def mixup(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup
    Args:
        x: input batch
        y: label batch

    Returns:
        mixed_x: mixed input batch
        y: label batch
        shuffled_y: shuffled label batch
        lam: lambda value
    """
    rand_index = torch.randperm(x.size(0)).to(x.device)
    shuffled_x = x[rand_index]
    shuffled_y = y[rand_index]

    lam = np.random.uniform(0.0, 1.0)
    x_mix = lam * x + (1 - lam) * shuffled_x
    return x_mix, y, shuffled_y, lam


def made_spec_augment_func(
    time_mask_param: int, freq_mask_param: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Make Spec-Augment function

    Args:
        time_mask_param: time masking parameter. maximum possible length of the mask.
                            Indices uniformly sampled from [0, time_mask_param]. max is n_frames

        freq_mask_param: frequency masking parameter. maximum possible length of the mask.
                            Indices uniformly sampled from [0, freq_mask_param]. max is height.

    Returns:
        spec_augment_func: Spec-Augment function. torch.Tensor -> torch.Tensor

    References:
    [1]
    https://arxiv.org/abs/1904.08779
    [2]
    https://pytorch.org/audio/stable/transforms.html
    """
    return nn.Sequential(
        TAT.TimeMasking(time_mask_param=time_mask_param),
        TAT.FrequencyMasking(freq_mask_param=freq_mask_param),
    )
