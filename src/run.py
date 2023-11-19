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
            # dl = build_dataloader_v3(self.dataconfig, fold, "valid", debug)
            dl = dataset.init_dataloader("valid", self.dataconfig)
            return dl
        else:
            # dl = build_dataloader_v3(self.dataconfig, fold, "test", debug)
            dl = dataset.init_dataloader("test", self.dataconfig)
            return dl

    def _init_model(self, config, weight_path: str | Path) -> nn.Module:
        model = build_model(config)
        logger.info("load: %s", weight_path)
        state_dict = torch.load(weight_path)
        print(model.load_state_dict(state_dict))
        model = model.to(self.device).eval()
        return model

    def _make_pred_v3(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        device = self.device
        # (BS, seq_len, n_features)
        X = batch[0].to(device, non_blocking=True)
        pred = torch.zeros(*X.shape[:2], 2).to(device, non_blocking=True)
        seq_len = X.shape[1]
        # Window sliding inference
        h = None
        for i in range(0, seq_len, chunk_size):
            ch_s = i
            ch_e = min(pred.shape[1], i + chunk_size)
            x_chunk = X[:, ch_s:ch_e, :].to(device, non_blocking=True)
            # logits = model(x_chunk, None, None)
            # logits, h = model(x_chunk, None)  # MultiResidualRNN
            logits, h = model(x_chunk, h)  # MultiResidualRNN
            h = [h_.detach() for h_ in h]
            pred[:, ch_s:ch_e] = logits.detach()
        return pred

    def _make_sub_v3(
        self,
        models: list[nn.Module],
        dl: DataLoader,
        criterion: Callable | None,
        losses: AverageMeter | None,
        # -- Additional params
        chunk_size: int,
        min_interval: int = 30,
    ) -> dict[str, np.ndarray | float | pd.DataFrame]:
        print("Infer ChunkSize => ", chunk_size)
        pbar = tqdm(enumerate(dl), total=len(dl), dynamic_ncols=True, leave=True)
        onset_losses = AverageMeter("onset_loss")
        wakeup_losses = AverageMeter("wakeup_loss")
        total_sub = pd.DataFrame()
        for i, batch in pbar:
            with torch.inference_mode():
                # (BS, seq_len, 2)
                sid = batch[2]
                pred = []
                for i, model in enumerate(models):
                    if isinstance(model, SleepTransformer):
                        pred_ = _infer_for_transformer(model, batch[0], self.device)

                    else:
                        pred_ = self._make_pred_v3(model, batch, chunk_size)
                    pred.append(pred_.detach().cpu().float())

                pred = torch.concat(pred)
                pred = pred.mean(0).unsqueeze(0)

                if criterion is not None and losses is not None:
                    normalized_pred = mean_std_normalize_label(pred)
                    y = batch[1].to(normalized_pred.device, non_blocking=True)
                    loss = criterion(normalized_pred, y)
                    loss_onset = (
                        criterion(normalized_pred[..., 0], y[..., 0]).detach().cpu()
                    )
                    if not torch.isnan(loss_onset):
                        onset_losses.update(loss_onset.item())

                    loss_wakeup = (
                        criterion(normalized_pred[..., 1], y[..., 1]).detach().cpu()
                    )
                    if not torch.isnan(loss_wakeup):
                        wakeup_losses.update(loss_wakeup.item())

                    losses.update(loss.item())
                    pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))

                # (BS, seq_len, 2) -> (seq_len, 2)
                pred = pred.detach().cpu().float().numpy().reshape(-1, 2)
                days = len(pred) // (17280 / 12)

                # Make scores of onset/wakeup
                # idxから前後min_intervalの中で最大の値を取る
                # これがidxの値と同じならば、そのidxはonset/wakeupの候補となる
                # len(pred) // (17280 / 12) で、predの長さを日数に変換
                score_onset = np.zeros(len(pred), np.float32)
                score_wakeup = np.zeros(len(pred), np.float32)
                for idx in range(len(pred)):
                    p_onset = pred[idx, 0]
                    p_wakeup = pred[idx, 1]

                    max_p_interval_onset = max(
                        pred[max(0, idx - min_interval) : idx + min_interval, 0]
                    )
                    max_p_interval_wakeup = max(
                        pred[max(0, idx - min_interval) : idx + min_interval, 1]
                    )
                    if p_onset == max_p_interval_onset:
                        score_onset[idx] = p_onset
                    if p_wakeup == max_p_interval_wakeup:
                        score_wakeup[idx] = p_wakeup

                # Select event(onset/wakeup) step index
                candidates_onset = np.argsort(score_onset)[-max(1, round(days)) :]
                candidates_wakeup = np.argsort(score_wakeup)[-max(1, round(days)) :]

                step = pd.DataFrame(dict(step=batch[3].numpy().reshape(-1)))

                # Make onset
                # 1 sample per minite
                downsample_factor = 12
                onset = step.iloc[
                    np.clip(candidates_onset * downsample_factor, 0, len(step) - 1)
                ].astype(np.int32)
                if isinstance(onset, pd.Series):
                    onset = onset.to_frame().T
                onset["event"] = "onset"
                onset["score"] = score_onset[candidates_onset]
                onset["series_id"] = sid[0]

                # Make wakeup
                wakeup = step.iloc[
                    np.clip(candidates_wakeup * downsample_factor, 0, len(step) - 1)
                ].astype(np.int32)
                if isinstance(wakeup, pd.Series):
                    wakeup = wakeup.to_frame().T
                wakeup["event"] = "wakeup"
                wakeup["score"] = score_wakeup[candidates_wakeup]
                wakeup["series_id"] = sid[0]

                total_sub = pd.concat([total_sub, onset, wakeup], axis=0)

        total_sub = total_sub.sort_values(["series_id", "step"]).reset_index(drop=True)
        total_sub["row_id"] = total_sub.index.astype(int)
        total_sub["score"] = total_sub["score"].fillna(total_sub["score"].mean())
        total_sub = total_sub[["row_id", "series_id", "step", "event", "score"]]
        submission = total_sub[["row_id", "series_id", "step", "event", "score"]]

        outputs = {}
        outputs["submission"] = submission
        if self.is_val and losses is not None:
            outputs["loss"] = losses.avg
            outputs["onset_loss"] = onset_losses.avg
            outputs["wakeup_loss"] = wakeup_losses.avg

        return outputs

    def _make_sub_for_seg(
        self,
        models: Sequence[nn.Module],
        dl: DataLoader,
        loss_fn: Callable | None,
        loss_monitor: AverageMeter | None,
        use_amp: bool = False,
        duration: int = 10,
        score_thr: float = 0.5,
        distance: int = 24 * 60 * 12,
    ) -> dict[str, pl.DataFrame]:
        print("Infer Duration => ", duration)
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

        preds = np.concatenate(preds)
        print("preds.shape: ", preds.shape, "len(keys): ", len(keys))
        assert preds.shape[0] == len(keys), f"{preds.shape=}, {len(keys)=}"
        sub_df = my_utils.post_process_for_seg(
            keys, preds[:, :, [1, 2]], score_thr=score_thr, distance=distance
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
