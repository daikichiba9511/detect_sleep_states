import importlib
import pprint
import time
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.tools import AverageMeter, LossFunc, Scheduler, train_one_fold
from src.utils import LoggingUtils, get_class_vars, seed_everything

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
) -> dict[str, float | int]:
    seed_everything(seed)
    model = model.to(device).train()
    scheduler.step(epoch)
    start_time = time.time()
    losses = AverageMeter("loss")

    max_chunk_size = 24 * 60 * 100

    pbar = tqdm(
        enumerate(dl_train), total=len(dl_train), dynamic_ncols=True, leave=True
    )
    for _, batch in pbar:
        with autocast(enabled=use_amp, dtype=torch.bfloat16):
            series = torch.concat(batch[0]).float().to(device, non_blocking=True)
            labels = torch.concat(batch[1]).float().to(device, non_blocking=True)
            seq_ln = series.shape[0]
            h = None
            for start in range(0, seq_ln, max_chunk_size):
                label = labels[start : start + max_chunk_size]
                outs, h = model(series[start : start + max_chunk_size], h)
                loss = criterion(outs, label)
                h = [h_.detach() for h_ in h]

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


def valid_one_epoch_v2(
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

    max_chunk_size = 24 * 60 * 100

    pbar = tqdm(
        enumerate(dl_valid), total=len(dl_valid), dynamic_ncols=True, leave=True
    )
    for _, batch in pbar:
        series = torch.concat(batch[0]).float().to(device, non_blocking=True)
        labels = torch.concat(batch[1]).float().to(device, non_blocking=True)

        with torch.no_grad():
            seq_ln = series.shape[0]
            h = None
            for start in range(0, seq_ln, max_chunk_size):
                label = labels[start : start + max_chunk_size]
                outs, h = model(series[start : start + max_chunk_size], h)
                loss = criterion(outs, label)
                h = [h_.detach() for h_ in h]

                loss = criterion(outs, labels)

                losses.update(loss.item())
                pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))

    result = {
        "epoch": epoch,
        "loss": losses.avg,
        "elapsed_time": time.time() - start_time,
    }
    return result


def main(config: str, fold: int, debug: bool) -> None:
    cfg = importlib.import_module(f"src.configs.{config}").Config
    log_fp = cfg.output_dir / f"{config}_fold{fold}.log"
    LoggingUtils.add_file_handler(LOGGER, log_fp)

    cfg_map = get_class_vars(cfg)
    LOGGER.info(f"Fold: {fold}\n {pprint.pformat(cfg_map)}")

    train_one_fold(
        cfg,
        fold,
        debug,
        train_one_epoch=train_one_epoch_v2,
        valid_one_epoch=valid_one_epoch_v2,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", type=str, default="exp000")
    main(**vars(parser.parse_args()))
