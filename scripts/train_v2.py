import importlib
import pprint
import time
import warnings
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import mean_std_normalize_label
from src.tools import AverageMeter, LossFunc, Scheduler, get_lr, train_one_fold, mixup
from src.utils import (
    LoggingUtils,
    get_class_vars,
    seed_everything,
    get_commit_head_hash,
)

warnings.filterwarnings("ignore")
INFO = 20
LOGGER = LoggingUtils.get_stream_logger(INFO)


def train_one_epoch_v3(
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
    # -- additional params
    chunk_size: int = 100,
) -> dict[str, float | int]:
    seed_everything(seed)
    model = model.to(device).train()
    scheduler.step(epoch)
    start_time = time.time()
    losses = AverageMeter("loss")

    pbar = tqdm(
        enumerate(dl_train), total=len(dl_train), dynamic_ncols=True, leave=True
    )
    for _, batch in pbar:
        with autocast(enabled=use_amp, dtype=torch.bfloat16):
            X = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            mixed = False
            if np.random.rand() < 0.5:
                X, y, y_mix, lam = mixup(X, y)
                mixed = True
            else:
                y_mix, lam = None, None

            pred, _ = model(X, None)  # MultiResidualBiGRU exp007
            normalized_pred = mean_std_normalize_label(pred)

            loss = criterion(normalized_pred, y)
            if mixed and y_mix is not None and lam is not None:
                loss_mixed = criterion(normalized_pred, y_mix)
                loss *= lam
                loss += loss_mixed * (1 - lam)

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
        "lr": get_lr(scheduler, epoch),
    }
    return result


def valid_one_epoch_v3(
    epoch: int,
    model: nn.Module,
    dl_valid: DataLoader,
    device: torch.device,
    criterion: LossFunc,
    seed: int = 42,
    # -- additional params
    chunk_size: int = 100,
) -> dict[str, float | int]:
    seed_everything(seed)
    model.eval()
    losses = AverageMeter("loss")
    onset_losses = AverageMeter("onset_loss")
    wakeup_losses = AverageMeter("wakeup_loss")
    onset_pos_only_losses = AverageMeter("onset_pos_only_loss")
    wakeup_pos_only_losses = AverageMeter("wakeup_pos_only_loss")
    start_time = time.time()

    pbar = tqdm(
        enumerate(dl_valid), total=len(dl_valid), dynamic_ncols=True, leave=True
    )
    for _, batch in pbar:
        with torch.inference_mode():
            # (BS, seq_len, n_features)
            X = batch[0].to(device, non_blocking=True)
            # (BS, seq_len, 2)
            y = batch[1].to(device, non_blocking=True)
            pred, _ = model(X, None)  # MultiResidualBiGRU exp007
            # logits, h = model(x_chunk, h)  # MultiResidualBiGRU
            normalized_pred = mean_std_normalize_label(pred)
            loss = criterion(normalized_pred, y)

            loss_onset = criterion(normalized_pred[..., 0], y[..., 0]).detach().cpu()
            if not torch.isnan(loss_onset):
                onset_losses.update(loss_onset.item())

            loss_wakeup = criterion(normalized_pred[..., 1], y[..., 1]).detach().cpu()
            if not torch.isnan(loss_wakeup):
                wakeup_losses.update(loss_wakeup.item())

            losses.update(loss.item())
            pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))

    result = {
        "epoch": epoch,
        "loss": losses.avg,
        "onset_loss": onset_losses.avg,
        "wakeup_loss": wakeup_losses.avg,
        "onset_pos_only_loss": onset_pos_only_losses.avg,
        "wakeup_pos_only_loss": wakeup_pos_only_losses.avg,
        "elapsed_time": time.time() - start_time,
    }
    return result


def main(config: str, fold: int, debug: bool) -> None:
    cfg = importlib.import_module(f"src.configs.{config}").Config
    cfg.model_save_path = cfg.output_dir / (cfg.model_save_name + f"{fold}.pth")
    log_fp = cfg.output_dir / f"{config}_fold{fold}.log"
    LoggingUtils.add_file_handler(LOGGER, log_fp)
    commit_hash = get_commit_head_hash()

    cfg_map = get_class_vars(cfg)
    LOGGER.info(f"fold: {fold}, debug: {debug}, commit_hash: {commit_hash}")
    LOGGER.info(f"Fold: {fold}\n {pprint.pformat(cfg_map)}")
    train_one_fold(
        cfg,
        fold,
        debug,
        partial(train_one_epoch_v3, chunk_size=cfg.train_chunk_size),
        partial(valid_one_epoch_v3, chunk_size=cfg.infer_chunk_size),
    )
    LOGGER.info(f"Fold {fold} training has finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", type=str, default="exp000")
    main(**vars(parser.parse_args()))
