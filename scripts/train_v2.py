import importlib
import pprint
import time
import warnings
from functools import partial
from logging import INFO

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import models as my_models
from src.dataset import mean_std_normalize_label
from src.tools import AverageMeter, LossFunc, Scheduler, get_lr, mixup, train_one_fold
from src.utils import (
    LoggingUtils,
    get_class_vars,
    get_commit_head_hash,
    seed_everything,
)

warnings.filterwarnings("ignore")


# for performance, but you need more accuracy, set it to False
torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
torch.backends.cudnn.allow_tf32 = True  # type: ignore

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
    use_amp: bool,
    seed: int = 42,
    num_grad_accum: int = 1,
    # -- additional params
) -> dict[str, float | int]:
    seed_everything(seed)
    model = model.to(device).train()
    scheduler.step(epoch)
    start_time = time.time()
    losses = AverageMeter("loss")

    pbar = tqdm(
        enumerate(dl_train), total=len(dl_train), dynamic_ncols=True, leave=True
    )
    for batch_idx, batch in pbar:
        with autocast(enabled=use_amp, dtype=torch.bfloat16):
            X = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            mixed = False
            if np.random.rand() < 0.5:
                X, y, y_mix, lam = mixup(X, y)
                mixed = True
            else:
                y_mix, lam = None, None

            # pred, _ = model(X, None)  # MultiResidualBiGRU exp007
            pred = model(X, training=True)  # SleepTransformer
            normalized_pred = mean_std_normalize_label(pred)

            loss = criterion(normalized_pred, y)
            if mixed and y_mix is not None and lam is not None:
                loss_mixed = criterion(normalized_pred, y_mix)
                loss *= lam
                loss += loss_mixed * (1 - lam)

            scaler.scale(loss).backward()  # type: ignore

            if (batch_idx + 1) % num_grad_accum == 0:
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
    use_amp: bool = False,
    # -- additional params
) -> dict[str, float | int]:
    seed_everything(seed)
    model = model.to(device).eval()
    losses = AverageMeter("loss")
    onset_losses = AverageMeter("onset_loss")
    wakeup_losses = AverageMeter("wakeup_loss")
    onset_pos_only_losses = AverageMeter("onset_pos_only_loss")
    wakeup_pos_only_losses = AverageMeter("wakeup_pos_only_loss")
    start_time = time.time()

    def _infer_for_transformer(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        chunk_size = 7200
        # (BS, seq_len, n_features)
        X = X.to(device, non_blocking=True)
        _, seq_len, n_features = X.shape
        X = X.squeeze(0)

        # print(f"1 {X.shape=}")
        if X.shape[0] % chunk_size != 0:
            X = torch.concat(
                [
                    X,
                    torch.zeros(
                        (chunk_size - len(X) % chunk_size, n_features),
                        device=X.device,
                    ),
                ],
            )

        # print(f"2 {X.shape=}")
        X_chunk = X.view(X.shape[0] // chunk_size, chunk_size, n_features)
        # print(f"3 {X.shape=}")
        # (BS, seq_len, 2)
        pred = model(X_chunk, training=False)
        pred = pred.reshape(-1, 2)[:seq_len].unsqueeze(0)
        return pred

    pbar = tqdm(
        enumerate(dl_valid), total=len(dl_valid), dynamic_ncols=True, leave=True
    )
    for _, batch in pbar:
        with torch.no_grad(), autocast(enabled=use_amp, dtype=torch.bfloat16):
            # (BS, seq_len, n_features)
            X = batch[0].to(device, non_blocking=True)
            # (BS, seq_len, 2)
            y = batch[1].to(device, non_blocking=True)

            if isinstance(model, my_models.SleepTransformer):
                pred = _infer_for_transformer(model, X)
            else:
                pred, _ = model(X, None)  # MultiResidualBiGRU exp007

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


def main(config: str, fold: int, debug: bool, model_compile: bool = False) -> None:
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
        partial(
            train_one_epoch_v3,
            num_grad_accum=cfg.num_grad_accum,
        ),
        partial(valid_one_epoch_v3),
        model_compile=model_compile,
        # compile_mode="max-autotune",
        compile_mode="default",
    )
    LOGGER.info(f"Fold {fold} training has finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", type=str, default="exp000")
    parser.add_argument("--model_compile", action="store_true")
    args = parser.parse_args()
    main(
        config=args.config,
        fold=args.fold,
        debug=args.debug,
        model_compile=False,
    )
