import os
import random
import time
from datetime import datetime, timedelta
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Any, Callable, ClassVar, Final, Literal, Sequence, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets import build_dataloader
from src.losses import build_criterion
from src.models import build_model

logger = getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""

    val: float | int
    avg: float | int
    sum: float | int
    count: int
    rows: list[float | int]

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __str__(self) -> str:
        return f"Metrics {self.name}: Avg {self.avg}"

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.rows: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        if value in [np.nan, np.inf, -np.inf, float("inf"), float("-inf")]:
            logger.info("Skip nan or inf value")
            return None
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.rows.append(value)

    def to_dict(self) -> dict[str, list[float | int] | str | float]:
        return {
            "name": self.name,
            "avg": self.avg,
            "row_values": self.rows,
        }


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        direction: Literal["maximize", "minimize"],
        delta: float = 0.0,
    ) -> None:
        if direction not in ["maximize", "minimize"]:
            raise ValueError(f"{direction = }")

        self._patience = patience
        self._direction = direction
        self._counter = 0
        self._best_score = float("-inf") if direction == "maximize" else float("inf")
        self._delta = delta

    def _is_improved(self, score: float, best_score: float) -> bool:
        if self._direction == "maximize":
            return score + self._delta > best_score
        else:
            return score + self._delta < best_score

    def _save(self, model: nn.Module, save_path: Path) -> None:
        state = model.state_dict()
        torch.save(state, save_path)
        logger.info(f"Saved model to {save_path}")

    def check(self, score: float, model: nn.Module, save_path: Path) -> None:
        if self._is_improved(score, self._best_score):
            logger.info(f"Score improved from {self._best_score} to {score}")
            self._best_score = score
            self._counter = 0
            self._save(model, save_path)
        else:
            self._counter += 1
            logger.info(
                f"EarlyStopping counter: {self._counter} out of {self._patience}. "
                + f"best: {self._best_score}"
            )

    @property
    def is_early_stop(self) -> bool:
        return self._counter >= self._patience

    @property
    def best_score(self) -> float:
        return self._best_score


class MetricsMonitor:
    def __init__(self, metrics: Sequence[str]) -> None:
        self.metrics = metrics
        self._metrics_df = pd.DataFrame(columns=["epoch", *metrics])

    def update(self, metrics: dict[str, float | int]) -> None:
        self._metrics_df = self._metrics_df.append(metrics, ignore_index=True)

    def show(self) -> None:
        logger.info(self._metrics_df)

    def plot(
        self,
        save_path: Path,
        col: str | Sequence[str],
        figsize: tuple[int, int] = (8, 6),
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        assert isinstance(ax, plt.Axes)
        assert isinstance(fig, plt.Figure)
        if isinstance(col, str):
            col = [col]
        for c in col:
            data = self._metrics_df[c].to_numpy()
            ax.plot(data, label=c)
        ax.set_xlabel("epoch")
        ax.legend()
        fig.savefig(save_path)

    def save(self, save_path: Path, fold: int) -> None:
        self._metrics_df["fold"] = fold
        self._metrics_df.to_csv(save_path, index=False)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.autograd.anomaly_mode.set_detect_anomaly(False)


LossFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train_one_epoch(
    fold: int,
    epoch: int,
    model: nn.Module,
    dl_train: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    criterion: LossFunc,
    device: torch.device,
    seed: int = 42,
    use_amp: bool = False,
) -> dict[str, float]:
    seed_everything(seed)
    model.train()
    scheduler.step(epoch)
    start_time = time.time()
    losses = AverageMeter("loss")

    pbar = tqdm(
        enumerate(dl_train), total=len(dl_train), dynamic_ncols=True, leave=True
    )
    for _, batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=use_amp, dtype=torch.float16):
            outs = model(images)
            loss = criterion(outs, labels)

            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item())

    result = {
        "epoch": epoch,
        "loss": losses.avg,
        "elapsed_time": time.time() - start_time,
    }
    return result


def valid_one_epoch(
    fold: int,
    epoch: int,
    model: nn.Module,
    dl_valid: DataLoader,
    device: torch.device,
    criterion: LossFunc,
) -> dict[str, float]:
    model.eval()
    losses = AverageMeter("loss")
    start_time = time.time()

    pbar = tqdm(
        enumerate(dl_valid), total=len(dl_valid), dynamic_ncols=True, leave=True
    )
    for _, batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.no_grad():
            outs = model(images)
            loss = criterion(outs, labels)

            losses.update(loss.item())

    result = {
        "epoch": epoch,
        "loss": losses.avg,
        "elapsed_time": time.time() - start_time,
    }
    return result


def train_one_fold(config, fold: int, debug: bool) -> None:
    logger.info(f"Start training fold{fold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer_params)
    scheduler = CosineLRScheduler(optimizer, **config.scheduler_params)
    criterion = build_criterion(config.criterion_name)
    dl_train = build_dataloader(config, fold, "train", debug)
    dl_valid = build_dataloader(config, fold, "valid", debug)

    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(**config.early_stopping_params)
    start_time = time.time()

    metrics_monitor = MetricsMonitor(["epoch", "loss"])
    for epoch in range(config.num_epochs):
        seed_everything(config.seed)
        logger.info(f"Start epoch {epoch}")
        train_result = train_one_epoch(
            fold=fold,
            epoch=epoch,
            model=model,
            dl_train=dl_train,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            device=device,
            seed=config.seed,
        )
        valid_result = valid_one_epoch(
            fold=fold,
            epoch=epoch,
            model=model,
            dl_valid=dl_valid,
            device=device,
            criterion=criterion,
        )
        metrics_monitor.update(
            {
                "epoch": epoch,
                "train/loss": train_result["loss"],
                "valid/loss": valid_result["loss"],
            }
        )
        metrics_monitor.show()

        score = valid_result["loss"]
        early_stopping.check(score, model, config.model_save_path)
        if early_stopping.is_early_stop:
            logger.info(
                f"Early Stopping at epoch {epoch}. best score is {early_stopping.best_score}"
            )
            break

    metrics_monitor.save(config.metrics_save_path, fold)
    logger.info(
        f"Training fold{fold} is done. elapsed time: {time.time() - start_time}"
    )
