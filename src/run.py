import importlib
from logging import getLogger
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import dataset
from src import utils as my_utils
from src.losses import build_criterion
from src.models import SleepTransformer, Spectrogram2DCNN, build_model
from src.tools import AverageMeter

logger = getLogger(__name__)


def plot_random_sample(keys, preds, labels, num_samples=1, num_chunks=10):
    import matplotlib.pyplot as plt

    # get series ids
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    # get random series
    random_series_ids = np.random.choice(unique_series_ids, num_samples)
    print(random_series_ids)

    for i, random_series_id in enumerate(random_series_ids):
        # get random series
        series_idx = np.where(series_ids == random_series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3)
        this_series_labels = labels[series_idx].reshape(-1, 3)

        # split series
        this_series_preds = np.split(this_series_preds, num_chunks)
        this_series_labels = np.split(this_series_labels, num_chunks)

        fig, axs = plt.subplots(num_chunks, 1, figsize=(20, 5 * num_chunks))
        if num_chunks == 1:
            axs = [axs]
        for j in range(num_chunks):
            this_series_preds_chunk = this_series_preds[j]
            this_series_labels_chunk = this_series_labels[j]

            assert isinstance(axs, np.ndarray)
            # get onset and wakeup idx
            onset_idx = np.nonzero(this_series_labels_chunk[:, 1])[0]
            wakeup_idx = np.nonzero(this_series_labels_chunk[:, 2])[0]

            axs[j].plot(this_series_preds_chunk[:, 0], label="pred_sleep")
            axs[j].plot(this_series_preds_chunk[:, 1], label="pred_onset")
            axs[j].plot(this_series_preds_chunk[:, 2], label="pred_wakeup")
            axs[j].vlines(
                onset_idx, 0, 1, label="onset", linestyles="dashed", color="C1"
            )
            axs[j].vlines(
                wakeup_idx, 0, 1, label="wakeup", linestyles="dashed", color="C2"
            )
            axs[j].set_ylim(0, 1)
            axs[j].set_title(f"series_id: {random_series_id} chunk_id: {j}")
            axs[j].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        fig.savefig(f"output/analysis/random_sample_{i}.png")  # type: ignore
        plt.close("all")


def _infer_for_transformer(
    model: nn.Module, X: torch.Tensor, device: torch.device
) -> torch.Tensor:
    chunk_size = 7200
    # (BS, seq_len, n_features)
    X = X.to(device, non_blocking=True)
    bs, seq_len, n_features = X.shape
    if bs != 1:
        raise ValueError(f"batch size must be 1, but {bs=}")
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


def _infer_for_seg(
    model: nn.Module, X: torch.Tensor, device: torch.device, duration: int = 10
) -> torch.Tensor:
    # (BS, seq_len, n_class)
    out = model(X)
    logits = out["logits"]
    resized_logits = TF.resize(
        logits.detach().cpu().sigmoid(),
        size=[duration, logits.shape[2]],
        antialias=False,
    )
    return resized_logits


class Runner:
    def __init__(
        self, configs, dataconfig, is_val: bool = False, device: str = "cuda"
    ) -> None:
        self.is_val = is_val
        self.configs = configs
        self.dataconfig = dataconfig
        self.device = torch.device(device)

    def _init_dl(self, debug: bool = False, fold: int = 0) -> DataLoader:
        logger.info("Debug mode: %s", debug)
        if self.is_val:
            print("\n###############################################\n")
            self.dataconfig.use_corrected_events = False
            # self.dataconfig.use_corrected_events_v2 = False
            self.dataconfig.use_corrected_events_v2 = False
            logger.info(my_utils.get_class_vars(self.dataconfig))
            print("\n###############################################\n")
            dl = dataset.init_dataloader("valid", self.dataconfig)
            return dl
        else:
            dl = dataset.init_dataloader("test", self.dataconfig)
            return dl

    def _init_model(self, config, weight_path: str | Path) -> nn.Module:
        model = build_model(config)
        logger.info("load: %s", weight_path)
        state_dict = torch.load(weight_path)
        print(model.load_state_dict(state_dict))
        model = model.to(self.device).eval()
        return model

    def _make_sub_for_seg(
        self,
        models: Sequence[nn.Module],
        dl: DataLoader,
        loss_fn: Callable | None,
        loss_monitor: AverageMeter | None,
        slide_size: int,
        use_amp: bool = False,
        duration: int = 10,
        score_thr: float = 0.5,
        distance: int = 24 * 60 * 12,
    ) -> dict[str, pl.DataFrame]:
        logger.info("Infer Duration => %s (slide_size %s)", duration, slide_size)
        # onset_losses = AverageMeter("onset_loss")
        # wakeup_losses = AverageMeter("wakeup_loss")
        # total_sub = pd.DataFrame()
        labels_all = []
        preds = []
        keys = []

        pbar = tqdm(enumerate(dl), total=len(dl), dynamic_ncols=True, leave=True)
        with torch.inference_mode(), autocast(use_amp):
            for _, batch in pbar:
                x = batch["feature"].to(self.device, non_blocking=True)
                keys.extend(batch["key"])
                logits_this_batch = []

                # Make preds for this batch
                pred_this_batch = torch.zeros((len(x), duration, 3))
                for model in models:
                    out = model(x)
                    # (BS, pred_length, n_class), pred_length = duration // downsample_rate
                    logits = out["logits"]
                    pred = logits.detach().cpu().float().sigmoid()
                    pred = TF.resize(
                        pred, size=[duration, pred.shape[2]], antialias=False
                    )
                    logits_this_batch.append(logits.detach().cpu().float())
                    pred_this_batch += pred

                pred_this_batch /= len(models)
                preds.append(pred_this_batch)

                # Only for validation
                if loss_fn is not None and loss_monitor is not None:
                    logits = torch.concat(logits_this_batch, dim=0).reshape(
                        len(models), x.shape[0], -1, 3
                    )
                    if len(models) != 1 and len(models) == logits.shape[0]:
                        logits = logits.mean(dim=0)
                    if len(models) == 1 and len(models) == logits.shape[0]:
                        logits = logits.squeeze(0)

                    labels = batch["label"].to(logits.device)
                    loss = loss_fn(logits, labels)
                    loss_monitor.update(loss.item())
                    labels_all.append(labels.detach().cpu().numpy())

        # preds: (bs * num_step, duration, 3)
        preds = np.concatenate(preds)
        print("preds.shape: ", preds.shape, "len(keys): ", len(keys))

        sub_df = my_utils.post_process_for_seg(
            keys,
            preds[:, :, [1, 2]],
            score_thr=score_thr,
            distance=distance,
            slide_size=slide_size,
        )

        if self.is_val:
            labels_all = np.concatenate(labels_all)
            plot_random_sample(keys, preds, labels_all, num_samples=5, num_chunks=10)

        outs = {"submission": sub_df}
        return outs

    def run(
        self,
        debug: bool = False,
        fold: int = 0,
        score_thr: float = 0.02,
        distance: int = 24 * 60 * 12,
    ) -> pd.DataFrame:
        models = []
        for config in self.configs:
            model = self._init_model(config, config.model_save_path)
            models.append(model)

        dl = self._init_dl(debug, fold)
        if self.is_val:
            criterion_type = "BCEWithLogitsLoss"
            logger.info("Criterion: %s", criterion_type)
            criterion = build_criterion(criterion_type)
            losses_meter = AverageMeter("loss")
        else:
            criterion = None
            losses_meter = None

        with my_utils.timer("start inference"):
            # outs = self._make_sub_v3(
            #     models, dl, criterion, losses_meter, chunk_size=chunk_size
            # )
            # TODO: durationとかをmodelごとに渡せるようにする, あまり意味ないか？
            outs = self._make_sub_for_seg(
                models,
                dl,
                criterion,
                losses_meter,
                use_amp=self.configs[0].use_amp,
                duration=self.configs[0].seq_len,
                score_thr=self.configs[0].postprocess_params["score_thr"],
                distance=self.configs[0].postprocess_params["distance"],
                slide_size=getattr(
                    self.configs[0], "slide_size", self.configs[0].seq_len
                ),
                # distance=24 * 60 * 8,
            )
            print(outs)

        # submission = self._make_sub(outs)
        submission = (
            outs["submission"]
            .to_pandas(use_pyarrow_extension_array=True)
            .reset_index(drop=True)
        )

        if self.is_val and losses_meter is not None:
            logger.info(f"fold: {fold}, loss: {losses_meter.avg:.5f}")
        return submission


if __name__ == "__main__":
    config = "exp005"
    config = importlib.import_module(f"src.configs.{config}").Config
    runner = Runner(configs=[config], dataconfig=config, is_val=False)
    submission = runner.run(debug=True, fold=0)
    print(submission)
