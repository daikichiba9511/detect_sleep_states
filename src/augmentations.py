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


def mixup_patch(
    x: torch.Tensor, y: torch.Tensor, n_parts: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup patches in a sequence

    Args:
        x: input batch. shape: (bs, n_channels, height, n_timesteps)
        y: label batch. shape: (bs, n_timesteps//downsample_rate, n_classes)
        n_parts: number of parts to split

    Returns:
        mixed_x: mixed input batch
        mixed_y: mixed label batch

    References:
    [1]
    https://www.kaggle.com/competitions/birdclef-2021/discussion/243463
    [2]
    https://www.kaggle.com/code/leehann/birdclef-21-2nd-place-model-train-0-66#GeM-and-Mix-up
    """
    bs, n_channels, height, n_timesteps = x.shape
    width = n_timesteps // n_parts


def rand_bbox1d(n_timesteps: int, lam: float) -> tuple[int, int]:
    """Random bounding box

    Args:
        n_timesteps: number of timesteps
        lam: lambda value

    Returns:
        cut_start: start index of cut
        cut_end: end index of cut
    """
    cut_rate = np.sqrt(1.0 - lam)
    cut_timesteps = int(n_timesteps * cut_rate)
    cut_start = np.random.randint(0, n_timesteps - cut_timesteps)
    cut_end = cut_start + cut_timesteps
    return cut_start, cut_end


def cutmix_1d(
    imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cutmix

    Args:
        imgs: input batch, (bs, n_channels, n_timesteps)
        labels: label batch, (bs, n_timesteps, n_classes)
        lam: lambda value

    Returns:
        mixed_imgs: mixed input batch
    """
    bs = imgs.size(0)
    random_index = torch.randperm(bs)

    shuffled_imgs = imgs[random_index]
    shuffled_labels = labels[random_index]

    lam = np.random.uniform(alpha, alpha)
    cut_start, cut_end = rand_bbox1d(imgs.size(2), lam)
    mixed_imgs = torch.concatenate(
        [
            imgs[:, :, :cut_start],
            shuffled_imgs[:, :, cut_start:cut_end],
            imgs[:, :, cut_end:],
        ],
        dim=2,
    )
    mixed_labels = torch.concatenate(
        [
            labels[:, :cut_start, :],
            shuffled_labels[:, cut_start:cut_end, :],
            labels[:, cut_end:, :],
        ],
        dim=1,
    )
    return mixed_imgs, mixed_labels


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
        # TAT.TimeStretch(fixed_rate=0.9),
        TAT.TimeMasking(time_mask_param=time_mask_param),
        TAT.FrequencyMasking(freq_mask_param=freq_mask_param),
    )


def make_aug_on_waveform() -> nn.Sequential:
    return nn.Sequential()
