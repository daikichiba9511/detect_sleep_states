import importlib
from logging import getLogger
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import build_dataloader
from src.losses import build_criterion
from src.models import build_model
from src.tools import AverageMeter

logger = getLogger(__name__)


# TODO: implement
class Runner:
    def __init__(
        self, config, dataconfig, is_val: bool = False, device: str = "cuda"
    ) -> None:
        self.is_val = is_val
        self.config = config
        self.dataconfig = dataconfig
        self.device = torch.device(device)

    def _init_dl(self, debug: bool = False, fold: int = 0) -> DataLoader:
        if self.is_val:
            dl = build_dataloader(self.dataconfig, fold, "valid", debug)
            return dl
        else:
            dl = build_dataloader(self.dataconfig, fold, "test", debug)
            return dl

    def _init_model(self) -> nn.Module:
        model = build_model(self.config)
        weight_path = self.config.model_save_path
        logger.info(f"load: {weight_path}")
        state_dict = torch.load(weight_path)
        print(model.load_state_dict(state_dict))
        model = model.to(self.device).eval()
        return model

    def run(self, debug: bool = False, fold: int = 0) -> pd.DataFrame:
        model = self._init_model()
        dl = self._init_dl(debug, fold)
        if self.is_val:
            criterion = build_criterion(self.config.criterion_type)
            losses = AverageMeter("loss")
        else:
            criterion = None
            losses = None

        series_ids = dl.dataset.series_ids  # type: ignore
        dfs = dl.dataset.data  # type: ignore
        submission = pd.DataFrame()

        max_chunk_size = 24 * 60 * 100
        min_interval = 30
        pbar = (
            tqdm(enumerate(dl), total=len(dl), dynamic_ncols=True, leave=True)
            if self.is_val
            else enumerate(dl)
        )
        for batch_idx, batch in pbar:
            series = torch.concat(batch[0]).float().to(self.device, non_blocking=True)
            if self.is_val:
                labels = (
                    torch.concat(batch[1]).float().to(self.device, non_blocking=True)
                )
            else:
                labels = None

            with torch.inference_mode():
                seq_ln = series.shape[0]
                h = None
                preds = torch.zeros((len(series), self.config.out_size))
                for start in range(0, seq_ln, max_chunk_size):
                    outs, h = model(series[start : start + max_chunk_size], h)
                    h = [h_.detach() for h_ in h]
                    preds[start : start + max_chunk_size] = outs.detach()

                    if (
                        self.is_val
                        and criterion is not None
                        and losses is not None
                        and labels is not None
                    ):
                        label = labels[start : start + max_chunk_size]
                        loss = criterion(outs, label)
                        losses.update(loss.item())
                        assert isinstance(pbar, tqdm)
                        pbar.set_postfix(dict(loss=f"{losses.avg:.5f}"))

            series_id = series_ids[batch_idx]
            preds = preds.cpu().softmax(dim=1).numpy()
            days = len(series) / (17280 / 12)
            score0 = np.zeros(len(preds), dtype=np.float16)
            score1 = np.zeros(len(preds), dtype=np.float16)
            for p_idx in range(len(preds)):
                pred_i_0 = preds[p_idx, 0]
                pred_i_1 = preds[p_idx, 1]
                pred_interval_0 = max(
                    preds[max(0, p_idx - min_interval) : p_idx + min_interval, 0]
                )
                pred_interval_1 = max(
                    preds[max(0, p_idx - min_interval) : p_idx + min_interval, 0]
                )
                if pred_i_0 == pred_interval_0:
                    score0[p_idx] = pred_i_0
                if pred_i_1 == pred_interval_1:
                    score1[p_idx] = pred_i_1
            candidates_onset = np.argsort(score0)[-max(1, round(days))]
            candidates_wakeup = np.argsort(score1)[-max(1, round(days))]
            onset = (
                dfs[batch_idx][["step"]]
                .iloc[np.clip(candidates_onset * 12, 0, len(dfs) - 1)]
                .astype(np.int32)
            )
            if not isinstance(onset, pd.DataFrame):
                onset = onset.to_frame().T
            onset["event"] = "onset"
            onset["series_id"] = series_id
            onset["score"] = score0[candidates_onset]
            wakeup = (
                dfs[batch_idx][["step"]]
                .iloc[np.clip(candidates_wakeup * 12, 0, len(dfs) - 1)]
                .astype(np.int32)
            )
            if not isinstance(wakeup, pd.DataFrame):
                wakeup = wakeup.to_frame().T
            wakeup["event"] = "wakeup"
            wakeup["series_id"] = series_id
            wakeup["score"] = score1[candidates_wakeup]

            submission = pd.concat([submission, onset, wakeup], axis=0)

        submission = submission.sort_values(["series_id", "step"]).reset_index(
            drop=True
        )
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
