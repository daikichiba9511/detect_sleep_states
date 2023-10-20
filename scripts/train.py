import importlib
import pprint
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.tools import AverageMeter, LossFunc, Scheduler, get_lr, train_one_fold
from src.utils import (
    LoggingUtils,
    get_class_vars,
    seed_everything,
    get_commit_head_hash,
)

warnings.filterwarnings("ignore")
INFO = 20
LOGGER = LoggingUtils.get_stream_logger(INFO)


def train_one_epoch_v2(
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
            input_data_num_array = batch["input_data_num_array"].to(
                device, non_blocking=True
            )
            input_data_mask_array = batch["input_data_mask_array"].to(
                device, non_blocking=True
            )
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            # (bs, seq_len, out_size)
            y = batch["y"].to(device, non_blocking=True)

            output = model(
                input_data_num_array,
                input_data_mask_array,
                attention_mask,
            )

            # y[..., 1] | y[...., 2] == 1のサンプルが一つは存在する
            if y[..., 1:].sum() != 0:
                # if True:
                # loss = criterion(
                #     output[input_data_mask_array == 1],
                #     y[input_data_mask_array == 1],
                # )
                margin = 5
                onset_use_index = y[input_data_mask_array == 1][..., 1] != 0
                true_indecies = torch.nonzero(onset_use_index)
                for idx in true_indecies:
                    start = max(0, idx.item() - margin)
                    end = min(len(onset_use_index) - 1, idx.item() + margin)
                    onset_use_index[start:end] = True
                loss_onset = criterion(
                    output[input_data_mask_array == 1][..., 1][onset_use_index],
                    y[input_data_mask_array == 1][..., 1][onset_use_index],
                )
                wakeup_use_index = y[input_data_mask_array == 1][..., 2] != 0
                true_indecies = torch.nonzero(wakeup_use_index)
                for idx in true_indecies:
                    start = max(0, idx.item() - margin)
                    end = min(len(wakeup_use_index) - 1, idx.item() + margin)
                    wakeup_use_index[start:end] = True
                loss_wakeup = criterion(
                    output[input_data_mask_array == 1][..., 2][wakeup_use_index],
                    y[input_data_mask_array == 1][..., 2][wakeup_use_index],
                )
                loss = loss_onset + loss_wakeup

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


def valid_one_epoch_v2(
    epoch: int,
    model: nn.Module,
    dl_valid: DataLoader,
    device: torch.device,
    criterion: LossFunc,
    seed: int = 42,
    # -- additional params
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
            input_data_num_array = batch["input_data_num_array"].to(
                device, non_blocking=True
            )
            input_data_mask_array = batch["input_data_mask_array"].to(
                device, non_blocking=True
            )
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            output = model(
                input_data_num_array,
                input_data_mask_array,
                attention_mask,
            )

            loss = criterion(
                output[input_data_mask_array == 1], y[input_data_mask_array == 1]
            )
            losses.update(loss.item())

            onset_loss = criterion(
                output[input_data_mask_array == 1][..., 1],
                y[input_data_mask_array == 1][..., 1],
            )
            onset_losses.update(onset_loss.item())

            wakeup_loss = criterion(
                output[input_data_mask_array == 1][..., 2],
                y[input_data_mask_array == 1][..., 2],
            )
            wakeup_losses.update(wakeup_loss.item())

            onset_pos_idx = y[input_data_mask_array == 1][..., 1] == 1
            onset_pos_only_loss = criterion(
                output[input_data_mask_array == 1][onset_pos_idx][..., 1],
                y[input_data_mask_array == 1][onset_pos_idx][..., 1],
            )
            if not torch.isnan(onset_pos_only_loss):
                onset_pos_only_losses.update(onset_pos_only_loss.item())

            wakeup_pos_idx = y[input_data_mask_array == 1][..., 2] == 1
            wakeup_pos_only_loss = criterion(
                output[input_data_mask_array == 1][wakeup_pos_idx][..., 2],
                y[input_data_mask_array == 1][wakeup_pos_idx][..., 2],
            )
            if not torch.isnan(wakeup_pos_only_loss):
                wakeup_pos_only_losses.update(wakeup_pos_only_loss.item())

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
    train_one_fold(cfg, fold, debug, train_one_epoch_v2, valid_one_epoch_v2)
    LOGGER.info(f"Fold {fold} training has finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", type=str, default="exp000")
    main(**vars(parser.parse_args()))
