import importlib
from logging import getLogger
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.tools import AverageMeter
from src.dataset import build_dataloader_v3, mean_std_normalize_label
from src.losses import build_criterion
from src.models import build_model, SleepTransformer
from src import utils as my_utils

logger = getLogger(__name__)


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
            dl = build_dataloader_v3(self.dataconfig, fold, "valid", debug)
            return dl
        else:
            dl = build_dataloader_v3(self.dataconfig, fold, "test", debug)
            return dl

    def _init_model(self, config, weight_path: str | Path) -> nn.Module:
        model = build_model(config)
        logger.info("load: %s", weight_path)
        state_dict = torch.load(weight_path)
        print(model.load_state_dict(state_dict))
        model = model.to(self.device).eval()
        return model

    def _make_preds(
        self,
        model: nn.Module,
        dl: DataLoader,
        criterion: Callable | None = None,
        losses: AverageMeter | None = None,
    ) -> dict[str, np.ndarray]:
        pbar = (
            tqdm(enumerate(dl), total=len(dl), dynamic_ncols=True, leave=True)
            if self.is_val
            else enumerate(dl)
        )
        outs = {
            "preds": [],
            "steps": [],
            "pred_use_array": [],
            "series_ids": [],
            "time_array": [],
        }
        for batch_idx, batch in pbar:
            with torch.inference_mode():
                input_data_num_array = batch["input_data_num_array"].to(
                    self.device, non_blocking=True
                )
                input_data_mask_array = batch["input_data_mask_array"].to(
                    self.device, non_blocking=True
                )
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )

                # print(input_data_num_array.shape)
                # print(input_data_mask_array.shape)
                # print(attention_mask.shape)

                logits = model(
                    input_data_num_array,
                    input_data_mask_array,
                    attention_mask,
                )
                if criterion is not None and losses is not None:
                    y = batch["y"].to(self.device, non_blocking=True)
                    # print(logits.shape, y.shape)
                    loss = criterion(
                        logits[input_data_mask_array == 1],
                        y[input_data_mask_array == 1],
                    )
                    losses.update(loss.item())
                    assert isinstance(pbar, tqdm)
                    pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))
                # (BS, seq_len, 3)
                pred = logits.detach().cpu().float().softmax(-1).numpy()

                outs["preds"].append(pred)
                outs["steps"].append(batch["steps"].numpy())
                outs["pred_use_array"].append(batch["pred_use_array"].numpy())
                outs["series_ids"].append(np.array(batch["series_ids"]))

        outputs = {}
        outputs["preds"] = np.concatenate(outs["preds"], axis=0)
        outputs["steps"] = np.concatenate(outs["steps"], axis=0)
        outputs["pred_use_array"] = np.concatenate(outs["pred_use_array"], axis=0)
        outputs["series_ids"] = np.concatenate(outs["series_ids"], axis=0)
        return outputs

    def _make_sub(self, outs: dict[str, np.ndarray]) -> pd.DataFrame:
        preds = outs["preds"]
        steps = outs["steps"]
        pred_use_array = outs["pred_use_array"]
        # TODO: series_idsがおかしい
        series_ids = outs["series_ids"]

        cnt = {}
        for i in range(len(series_ids)):
            if series_ids[i] not in cnt:
                cnt[series_ids[i]] = 0
            cnt[series_ids[i]] += 1
        print(cnt)

        preds_list = []
        for i in range(len(pred_use_array)):
            mask_ = pred_use_array[i] == 1
            preds_ = preds[i][mask_]
            steps_ = steps[i][mask_]
            series_ids_ = series_ids[i]
            df_ = pd.DataFrame()
            df_["step"] = steps_
            df_["prob_noevent"] = preds_[:, 0]
            df_["prob_onset"] = preds_[:, 1]
            df_["prob_wakeup"] = preds_[:, 2]
            df_["series_id"] = series_ids_
            preds_list.append(df_)

        df_preds = pd.concat(preds_list, axis=0)
        df_preds = df_preds.groupby(["series_id", "step"]).max().reset_index()

        df_preds["prob_noevent"] = df_preds["prob_noevent"].clip(0.0005, 0.9995)
        df_preds["prob_onset"] = df_preds["prob_onset"].clip(0.005, 0.9995)
        df_preds["prob_wakeup"] = df_preds["prob_wakeup"].clip(0.005, 0.9995)

        print(df_preds)
        print(df_preds.groupby("series_id").mean())
        print(df_preds.groupby("series_id").max())
        print("#########################################")
        print(df_preds[["prob_noevent", "prob_onset", "prob_wakeup"]][:10])
        print(
            df_preds["prob_noevent"].max(),
            df_preds["prob_noevent"].min(),
            df_preds["prob_noevent"].mean(),
        )
        print(
            df_preds["prob_onset"].max(),
            df_preds["prob_onset"].min(),
            df_preds["prob_onset"].mean(),
        )
        print(
            df_preds["prob_wakeup"].max(),
            df_preds["prob_wakeup"].min(),
            df_preds["prob_wakeup"].mean(),
        )
        print("#########################################")

        onse_thr = 0.95
        wake_thr = 0.95

        # onset event
        df_preds["pred_onset"] = (df_preds["prob_onset"] > onse_thr).astype(int)
        df_preds_onset = df_preds[df_preds["pred_onset"] == 1]
        # assert df_preds_onset.shape[0] > 0, f"pred_onset: {df_preds_onset.shape}"
        df_preds_onset.loc[:, "event"] = "onset"
        df_preds_onset = df_preds_onset[["series_id", "step", "event", "prob_onset"]]
        df_preds_onset.columns = ["series_id", "step", "event", "score"]

        # wakeup event
        df_preds["pred_wakeup"] = (df_preds["prob_wakeup"] > wake_thr).astype(int)
        df_preds_wakeup = df_preds[df_preds["pred_wakeup"] == 1]
        # assert df_preds_wakeup.shape[0] > 0, f"pred_wakeup: {df_preds_wakeup.shape}"
        df_preds_wakeup.loc[:, "event"] = "wakeup"
        df_preds_wakeup = df_preds_wakeup[["series_id", "step", "event", "prob_wakeup"]]
        df_preds_wakeup.columns = ["series_id", "step", "event", "score"]

        submission = (
            pd.concat([df_preds_onset, df_preds_wakeup], axis=0)
            .sort_values(["series_id", "step"])
            .reset_index(drop=True)
        )

        print("############# SUBMISSION ##############")
        print(submission)
        print("#################################")

        submission["row_id"] = submission.index.astype(int)
        submission["score"] = submission["score"].fillna(submission["score"].mean())
        submission = submission[["row_id", "series_id", "step", "event", "score"]]
        return submission

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

    def run(self, debug: bool = False, fold: int = 0) -> pd.DataFrame:
        models = []
        for config in self.configs:
            model = self._init_model(config, config.model_save_path)
            models.append(model)

        dl = self._init_dl(debug, fold)
        if self.is_val:
            criterion_type = "MSELoss"
            logger.info("Criterion: %s", criterion_type)
            criterion = build_criterion(criterion_type)
            losses_meter = AverageMeter("loss")
        else:
            criterion = None
            losses_meter = None

        chunk_size = self.configs[0].infer_chunk_size

        with my_utils.timer("start inference"):
            outs = self._make_sub_v3(
                models, dl, criterion, losses_meter, chunk_size=chunk_size
            )

        # submission = self._make_sub(outs)
        submission = outs["submission"]
        assert isinstance(submission, pd.DataFrame)

        if self.is_val and losses_meter is not None:
            logger.info(f"fold: {fold}, loss: {losses_meter.avg:.5f}")
        return submission


if __name__ == "__main__":
    config = "exp005"
    config = importlib.import_module(f"src.configs.{config}").Config
    runner = Runner(configs=[config], dataconfig=config, is_val=False)
    submission = runner.run(debug=True, fold=0)
    print(submission)
