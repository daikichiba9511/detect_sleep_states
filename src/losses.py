from functools import partial
from typing import Callable

import torch


def bce_with_postive_only(logits, targets):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    loss *= targets.sum(1)  # 両方1立ってると重み付けすごくなりそう
    loss = loss.nanmean()
    if loss.isnan():
        return torch.tensor(1.0, device=logits.device)
    return loss


def bce_with_weighted_postive(
    logits, targets, weights: tuple[float, float, float] | None = None
):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    if targets.shape[-1] != 3:
        raise ValueError(f"Expected targets.shape[-1] == 3, got {targets.shape[1]}")
    loss_onset = loss[targets[..., 1] == 1].nanmean(1).nanmean()
    if loss_onset.isnan():
        loss_onset = torch.tensor(0.0, device=logits.device)
    loss_wakeup = loss[targets[..., 2] == 1].nanmean(1).nanmean()
    if loss_wakeup.isnan():
        loss_wakeup = torch.tensor(0.0, device=logits.device)
    loss_non_event = loss[targets[..., 0] == 1].nanmean(1).nanmean()

    # w_i = 1 / (freq_in_targets) to address class imbalance
    #  shape: (2, 2)
    # ┌────────┬───────┐
    # │ event  ┆ count │
    # │ ---    ┆ ---   │
    # │ str    ┆ u32   │
    # ╞════════╪═══════╡
    # │ wakeup ┆ 7254  │
    # │ onset  ┆ 7254  │
    # └────────┴───────┘
    if weights is None:
        num_series = 25545780
        num_onset = 7254
        num_wakeup = 7254
        w_onset = 1 / (num_onset / num_series)
        w_wakeup = 1 / (num_wakeup / num_series)
        w_non_event = 1 / ((num_series - (num_wakeup + num_onset)) / num_series)
    else:
        w_onset, w_wakeup, w_non_event = weights
    loss = w_onset * loss_onset
    loss += w_wakeup * loss_wakeup
    loss += w_non_event * loss_non_event
    return loss


def build_criterion(criterion_type: str) -> Callable:
    if criterion_type == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    elif criterion_type == "BCEWithLogitsLossWeightedPos":
        return partial(bce_with_weighted_postive, weights=(0.8, 0.8, 0.2))
    elif criterion_type == "MSELoss":
        return torch.nn.MSELoss()
    else:
        raise NotImplementedError
