from typing import Callable

import torch


def build_criterion(criterion_type: str) -> Callable:
    if criterion_type == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError
