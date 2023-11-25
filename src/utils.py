import json
import math
import os
import pathlib
import random
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Any, ClassVar, Sequence
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import polars as pl
import psutil
import torch
from scipy import signal

logger = getLogger(__name__)


class LoggingUtils:
    format: ClassVar[
        str
    ] = "[%(levelname)s] %(asctime)s - %(pathname)s : %(lineno)d: %(funcName)s: %(message)s"

    @classmethod
    def get_stream_logger(cls, level: int = INFO) -> Logger:
        logger = getLogger()
        logger.setLevel(level)

        handler = StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(Formatter(cls.format))
        logger.addHandler(handler)
        return logger

    @classmethod
    def add_file_handler(cls, logger: Logger, filename: str, level: int = INFO) -> None:
        handler = FileHandler(filename=filename)
        handler.setLevel(level)
        handler.setFormatter(Formatter(cls.format))
        logger.addHandler(handler)


def get_called_time() -> str:
    """Get current time in JST (Japan Standard Time = UTC+9)"""
    now = datetime.utcnow() + timedelta(hours=9)
    return now.strftime("%Y%m%d%H%M%S")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.autograd.anomaly_mode.set_detect_anomaly(False)


def get_class_vars(cls_obj: object) -> dict[str, Any]:
    return {k: v for k, v in cls_obj.__dict__.items() if not k.startswith("__")}


def measure_fn(logger=logger.info):
    def _timer(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = fn(*args, **kwargs)
            end_time = time.time()
            logger(f"{fn.__name__} done in {end_time - start_time:.4f} s")
            return result

        return wrapper

    return _timer


@contextmanager
def timer(name, log_fn=logger.info):
    t0 = time.time()
    yield
    log_fn(
        "[{name}] done in {duration:.3f} s".format(name=name, duration=time.time() - t0)
    )


def get_commit_head_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=True
    ).stdout.decode("utf-8")[:-1]


@contextmanager
def trace(title: str):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta_mem = m1 - m0
    sign = "+" if delta_mem >= 0 else "-"
    delta_mem = math.fabs(delta_mem)
    print(f"{title}: {m1:.2f}GB({sign}{delta_mem:.2f}GB):{time.time() - t0:.3f}s")


def pad_if_needed(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) == max_len:
        return x

    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)  # type: ignore


def load_series(path: pathlib.Path, key: str, fold: int) -> list[str]:
    if not path.exists():
        return []

    with path.open("r") as f:
        series = json.load(f)
    fold_series = series[fold]
    assert fold == fold_series["fold"]
    return fold_series[key]


def post_process_for_seg(
    keys: Sequence[str],
    preds: np.ndarray,
    score_thr: float = 0.01,
    distance: int = 5000,
) -> pl.DataFrame:
    """
    Args:
        keys: series_id + "_" + event_name
        preds: (n, duration, 2) array. 0: onset, 1: wakeup
    """
    if preds.shape[-1] != 2:
        raise ValueError(f"Invalid shape: {preds.shape}. check your infer.")

    series_ids = np.array([key.split("_")[0] for key in keys])
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = signal.find_peaks(
                this_event_preds, height=score_thr, distance=distance
            )[0]
            scores = this_event_preds[steps]
            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "event": event_name,
                        "step": step,
                        "score": score,
                    }
                )

    if not records:
        records.append(
            {
                "series_id": unique_series_ids[0],
                "event": "onset",
                "step": 0,
                "score": 0.0,
            },
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(
        ["row_id", "series_id", "event", "step", "score"]
    )
    return sub_df


def transformed_record_state(record_state: pl.DataFrame) -> pl.DataFrame:
    # cleasing
    before_len = len(record_state)
    print("Before cleasing:", before_len)
    record_state = record_state.filter(pl.col("step") != 0)
    print(
        "After cleasing:",
        len(record_state),
        ": the number of rows are reduced by",
        before_len - len(record_state),
    )
    print("**********Cleansed")
    print(record_state)

    print("**********Transformed")
    record_state = record_state.select(
        pl.col("series_id"),
        pl.col("night"),
        pl.when(pl.col("awake") == 1)
        .then(pl.lit("wakeup"))
        .when(pl.col("periodic") == 1)
        .then(pl.lit("periodic"))
        .otherwise(pl.lit("onset"))
        .alias("event"),
        pl.col("step"),
        pl.col("timestamp"),
    )

    # dfs = []
    # for series_id, series_df in record_state.groupby("series_id"):
    #     series_df = series_df.with_columns(
    #         pl.Series("night", np.arange(len(series_df)) // 2 + 1)
    #     )
    #     dfs.append(series_df)
    # df = pl.concat(dfs).sort(by=["series_id", "night"])

    return record_state


def replace_nonconsecutive_true_with_false(arr: np.ndarray) -> np.ndarray:
    """連続していないTrueをFalseに置き換える

    References:
    [1] https://www.kaggle.com/code/welshonionman/cmi-submit-edc250?scriptVersionId=151848507
    """
    arr_copy = arr.copy()

    # 最初の要素がTrueで、2番目の要素がFalseのとき、最初の要素をFalseにする
    if arr[0] and not arr[1]:
        arr_copy[0] = False

    # i-1とi+1がFalseのとき、iをFalseにする
    for i in range(1, len(arr) - 1):
        if not arr[i - 1] and not arr[i + 1]:
            arr_copy[i] = False

    # 最後の要素がTrueで、最後から2番目の要素がFalseのとき、最後の要素をFalseにする
    if arr[-1] and not arr[-2]:
        arr_copy[-1] = False

    return arr_copy


def exclude_periodic_feature(features: np.ndarray) -> np.ndarray:
    """周期的な特徴量を除外する

    Args:
        features: (n_timesteps, n_features)

    Returns:
        periodic_pos: (num_periodic_steps, )

    References:
    [1] https://www.kaggle.com/code/welshonionman/cmi-submit-edc250?scriptVersionId=151848507
    """
    # 17280 = 24 * 60 * 12 step/min
    days = (len(features) - 1) // 17280

    periodic = np.zeros((features.shape[0], 1))
    dummy = np.full_like(features, 10)
    for day in range(1, days + 1):
        shift_step = 17280 * day
        forward_shift = np.concatenate((dummy[-shift_step:], features[:-shift_step]))
        backward_shift = np.concatenate((features[shift_step:], dummy[:shift_step]))

        match_forward = np.all(forward_shift == features, axis=1)
        match_backward = np.all(backward_shift == features, axis=1)
        match_cond = np.expand_dims(np.logical_or(match_forward, match_backward), 1)
        periodic = np.logical_or(periodic, match_cond)

    periodic = replace_nonconsecutive_true_with_false(periodic).reshape(-1)
    periodic_pos = np.where(periodic)[0]
    return periodic_pos


def create_periodic_dict(series: pd.DataFrame) -> dict[str, np.ndarray]:
    """各series_idの周期的な部分を特定する

    Args:
        series: series_id, timestamp, anglez, enmo

    Returns:
        periodic_dict: series_idをキーとし、周期的な部分のstepを値とする辞書

    References:
    [1] https://www.kaggle.com/code/welshonionman/cmi-submit-edc250?scriptVersionId=151848507
    """

    series_ids = tqdm(series["series_id"].unique().tolist())
    periodic_dict = {}
    for series_id in series_ids:
        train_each_seriesid = series.query("series_id == @series_id")
        # train_each_seriesid.loc[:, "date_time"] = pd.to_datetime(
        #     train_each_seriesid["timestamp"], utc=True
        # )
        features = train_each_seriesid[["anglez", "enmo"]].to_numpy()
        periodic = exclude_periodic_feature(features)
        periodic_dict[series_id] = periodic
    return periodic_dict


def remove_periodic(
    df: pd.DataFrame, periodic_dict: dict[str, np.ndarray]
) -> pd.DataFrame:
    """周期的な部分に該当する行を削除する

    References:
    [1] https://www.kaggle.com/code/welshonionman/cmi-submit-edc250?scriptVersionId=151848507
    """
    df_ = df.copy()
    df_["periodic"] = 0

    series_ids = sorted(df_["series_id"].unique().tolist())
    for series_id in series_ids:
        this_seriesid_df = df_.query("series_id == @series_id")

        for index, step in zip(this_seriesid_df.index, this_seriesid_df["step"]):
            if step in periodic_dict[series_id]:
                df_.loc[index, "periodic"] = 1

    df_ = df_.query("periodic == 0").drop("periodic", axis=1)
    return df_


if __name__ == "__main__":
    print(get_commit_head_hash())
