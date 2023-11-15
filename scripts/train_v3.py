import importlib
import pprint
import time
import warnings
from functools import partial
from logging import INFO

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad
from tqdm.auto import tqdm

from src import metrics, utils
from src import models as my_models
from src.dataset import mean_std_normalize_label
from src.tools import AverageMeter, LossFunc, Scheduler, get_lr, train_one_fold
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


def train_one_epoch_v4(
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
    mixup_raw_signal_prob: float = 0.0,
    mixup_prob: float = 0.0,
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
        with autocast(enabled=use_amp, dtype=torch.float16):
            X = batch["feature"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            # mixed = False
            # if np.random.rand() < 0.5:
            #     X, y, y_mix, lam = mixup(X, y)
            #     mixed = True
            # else:
            #     y_mix, lam = None, None

            do_mixup = np.random.rand() < mixup_prob
            do_mixup_raw_signal = np.random.rand() < mixup_raw_signal_prob
            out = model(
                X, y, do_mixup=do_mixup, do_mixup_raw_signal=do_mixup_raw_signal
            )  # Spectrogram2DCNN
            loss = out["loss"]

            # logits = out["logits"]
            # if mixed and y_mix is not None and lam is not None:
            #     loss_mixed = criterion(logits, y_mix)
            #     loss *= lam
            #     loss += loss_mixed * (1 - lam)

            if not torch.isnan(loss) or not torch.isinf(loss):
                scaler.scale(loss).backward()  # type: ignore

            if (batch_idx + 1) % num_grad_accum == 0 and not (
                torch.isnan(loss) or torch.isinf(loss)
            ):
                clip_grad.clip_grad_norm_(model.parameters(), 1.0)
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


def valid_one_epoch_v4(
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
    sleep_losses = AverageMeter("sleep_loss")
    onset_pos_only_losses = AverageMeter("onset_pos_only_loss")
    wakeup_pos_only_losses = AverageMeter("wakeup_pos_only_loss")
    bce = nn.BCEWithLogitsLoss()
    start_time = time.time()

    pbar = tqdm(
        enumerate(dl_valid), total=len(dl_valid), dynamic_ncols=True, leave=True
    )
    # valid_output = []
    for _, batch in pbar:
        with torch.no_grad(), autocast(enabled=use_amp, dtype=torch.float16):
            # (BS, n_features, seq_len)
            X = batch["feature"].to(device, non_blocking=True)
            # (BS, seq_len, 3)
            y = batch["label"].to(device, non_blocking=True)
            out = model(X, y)  # Spectrogram2DCNN
            # logits = out["logits"]
            loss = out["loss"]
            if not (torch.isnan(loss) or torch.isinf(loss)):
                losses.update(loss.item())
                pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))

            logits = out["logits"].detach()

            # Loss_sleep
            y_sleep = y[:, :, 0]
            y_sleep_pos_indices = y_sleep > 0
            loss_sleep = bce(
                logits[y_sleep_pos_indices][:, 0], y_sleep[y_sleep_pos_indices]
            )
            if not torch.isnan(loss_sleep):
                sleep_losses.update(loss_sleep.item())

            # Loss_onset
            y_onset = y[:, :, 1]
            y_onset_pos_indices = y_onset > 0
            loss_onset = bce(
                logits[y_onset_pos_indices][:, 1], y_onset[y_onset_pos_indices]
            )
            if not torch.isnan(loss_onset):
                onset_losses.update(loss_onset.item())

            # Loss_wakeup
            y_wakeup = y[:, :, 2]
            y_wakeup_pos_indices = y_wakeup > 0
            loss_wakeup = bce(
                logits[y_wakeup_pos_indices][:, 2], y_wakeup[y_wakeup_pos_indices]
            )
            if not torch.isnan(loss_wakeup):
                wakeup_losses.update(loss_wakeup.item())

            # resized_logits = TF.resize(
            #     logits.sigmoid().detach().cpu(),
            #     size=[duration, logits.shape[2]],
            #     antialias=False,
            # )
            # resized_y = TF.resize(
            #     y.detach().cpu(),
            #     size=[duration, logits.shape[2]],
            #     antialias=False,
            # )
            # valid_output.append(
            #     (
            #         resized_logits,
            #         resized_y,
            #         loss.detach().item(),
            #     )
            # )

    result = {
        "epoch": epoch,
        "loss": losses.avg,
        "onset_loss": onset_losses.avg,
        "wakeup_loss": wakeup_losses.avg,
        "sleep_loss": sleep_losses.avg,
        "onset_pos_only_loss": onset_pos_only_losses.avg,
        "wakeup_pos_only_loss": wakeup_pos_only_losses.avg,
        "elapsed_time": time.time() - start_time,
    }
    return result


def main(config: str, fold: int, debug: bool, model_compile: bool = False) -> None:
    cfg = importlib.import_module(f"src.configs.{config}").Config
    cfg.model_save_path = cfg.output_dir / (cfg.model_save_name + f"{fold}.pth")
    if debug:
        cfg.train_series = cfg.train_series[:5]
        cfg.sample_per_epoch = None

    log_fp = cfg.output_dir / f"{config}_fold{fold}.log"
    LoggingUtils.add_file_handler(LOGGER, log_fp)
    commit_hash = get_commit_head_hash()

    cfg_map = get_class_vars(cfg)
    LOGGER.info(
        f"running train_v3.py => fold: {fold}, debug: {debug}, commit_hash: {commit_hash}"
    )
    LOGGER.info(f"Fold: {fold}\n {pprint.pformat(cfg_map)}")
    train_one_fold(
        cfg,
        fold,
        debug,
        partial(
            train_one_epoch_v4,
            num_grad_accum=cfg.num_grad_accum,
            mixup_prob=cfg.mixup_prob,
            mixup_raw_signal_prob=getattr(cfg, "mixup_raw_signal_prob", 0.0),
        ),
        partial(valid_one_epoch_v4),
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
