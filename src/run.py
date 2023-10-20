import importlib
from logging import getLogger
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import build_dataloader_v2
from src.losses import build_criterion
from src.models import build_model
from src.tools import AverageMeter

logger = getLogger(__name__)


class Runner:
    def __init__(
        self, config, dataconfig, is_val: bool = False, device: str = "cuda"
    ) -> None:
        self.is_val = is_val
        self.config = config
        self.dataconfig = dataconfig
        self.device = torch.device(device)

    def _init_dl(self, debug: bool = False, fold: int = 0) -> DataLoader:
        print("Debug mode:", debug)
        if self.is_val:
            dl = build_dataloader_v2(
                self.dataconfig, fold, "valid", debug, use_cache=False
            )
            return dl
        else:
            dl = build_dataloader_v2(self.dataconfig, fold, "test", debug)
            return dl

    def _init_model(self) -> nn.Module:
        model = build_model(self.config)
        weight_path = self.config.model_save_path
        logger.info(f"load: {weight_path}")
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

    def run(self, debug: bool = False, fold: int = 0) -> pd.DataFrame:
        model = self._init_model()
        dl = self._init_dl(debug, fold)
        if self.is_val:
            criterion = build_criterion(self.config.criterion_type)
            losses = AverageMeter("loss")
        else:
            criterion = None
            losses = None

        submission = pd.DataFrame()
        outs = self._make_preds(model, dl, criterion, losses)

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

        if self.is_val and losses is not None:
            logger.info(f"fold: {fold}, loss: {losses.avg:.5f}")
        return submission


if __name__ == "__main__":
    config = "exp000"
    config = importlib.import_module(f"src.configs.{config}").Config
    runner = Runner(config=config, dataconfig=config, is_val=False)
    submission = runner.run(debug=True, fold=0)
    print(submission)
