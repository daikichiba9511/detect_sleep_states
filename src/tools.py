import time
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

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
from typing_extensions import TypeAlias

from src.dataset import build_dataloader
from src.losses import build_criterion
from src.models import build_model
from src.utils import seed_everything

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
        self._metrics_df = pd.DataFrame(columns=[*metrics])

    def update(self, metrics: dict[str, float | int]) -> None:
        epoch = metrics.pop("epoch")
        _metrics = pd.DataFrame(metrics, index=[epoch])
        self._metrics_df = pd.concat([self._metrics_df, _metrics], axis=0)

    def show(self, log_interval: int = 1) -> None:
        loggin_metrics = self._metrics_df.iloc[
            list(range(0, len(self._metrics_df), log_interval))
        ]
        logger.info(f"\n{loggin_metrics}")

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


LossFunc: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Scheduler: TypeAlias = torch.optim.lr_scheduler._LRScheduler | CosineLRScheduler


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dl_train: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler,
    scaler: GradScaler,
    criterion: LossFunc,
    device: torch.device,
    seed: int = 42,
    use_amp: bool = False,
) -> dict[str, float | int]:
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
            pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))

    result = {
        "epoch": epoch,
        "loss": losses.avg,
        "elapsed_time": time.time() - start_time,
    }
    return result


def create_checkpoints(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler,
    scaler: torch.cuda.amp.grad_scaler.GradScaler | None,
    score: float,
    epoch: int,
    save_weight_only: bool = False,
) -> dict:
    if save_weight_only:
        return {
            "model_state_dict": model.state_dict(),
        }
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "score": score,
        "epoch": epoch,
    }


def valid_one_epoch(
    epoch: int,
    model: nn.Module,
    dl_valid: DataLoader,
    device: torch.device,
    criterion: LossFunc,
    seed: int = 42,
) -> dict[str, float | int]:
    seed_everything(seed)
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
            pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))

    result = {
        "epoch": epoch,
        "loss": losses.avg,
        "elapsed_time": time.time() - start_time,
    }
    return result


class TrainConfig(Protocol):
    seed: int

    model_save_path: Path
    metrics_save_path: Path
    metrics_plot_path: Path

    criterion_type: str
    optimizer_params: dict[str, Any]
    scheduler_params: dict[str, Any]
    early_stopping_params: dict[str, Any]

    use_amp: bool
    num_epochs: int
    batch_size: int
    num_workers: int

    early_stopping_params: dict[str, Any]

    # Used in build_dataloader
    window_size: int
    train_series_path: str | Path
    train_events_path: str | Path
    test_series_path: str | Path

    # Used in build_model
    model_type: str
    input_size: int
    hidden_size: int
    out_size: int
    n_layers: int
    bidir: bool


def train_one_fold(
    config: TrainConfig,
    fold: int,
    debug: bool,
    train_one_epoch: Callable = train_one_epoch,
    valid_one_epoch: Callable = valid_one_epoch,
    log_interval: int = 1,
) -> None:
    logger.info(f"Start training fold{fold}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer_params)
    scheduler = CosineLRScheduler(optimizer, **config.scheduler_params)
    criterion = build_criterion(config.criterion_type)
    dl_train = build_dataloader(config, fold, "train", debug)
    dl_valid = build_dataloader(config, fold, "valid", debug)

    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(**config.early_stopping_params)
    start_time = time.time()

    metrics_monitor = MetricsMonitor(["train/loss", "valid/loss"])
    for epoch in range(config.num_epochs):
        seed_everything(config.seed)
        logger.info(f"Start epoch {epoch}")
        train_result = train_one_epoch(
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
        if epoch % log_interval == 0:
            metrics_monitor.show(log_interval=log_interval)

        score = valid_result["loss"]
        early_stopping.check(score, model, config.model_save_path)
        if early_stopping.is_early_stop:
            logger.info(
                f"Early Stopping at epoch {epoch}. best score is {early_stopping.best_score}"
            )
            break

    metrics_monitor.save(config.metrics_save_path, fold)
    metrics_monitor.plot(config.metrics_plot_path, col=["train/loss", "valid/loss"])
    logger.info(
        f"Training fold{fold} is done. elapsed time: {time.time() - start_time}"
    )
