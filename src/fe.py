from logging import getLogger
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = getLogger(__name__)


def mean_std_normalize(x: np.ndarray, eps: float = 1e-4) -> pl.Series:
    return (x - x.mean()) / (x.std() + eps)


def robust_normalize(x: np.ndarray) -> np.ndarray:
    scaler = RobustScaler()
    return scaler.fit_transform(x)


def standard_normalize(x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def make_sequence_chunks(
    df_series: pl.DataFrame,
    seq_len: int,
    shift_size: int,
    offset_size: int,
    normalize_type: str = "robust",
    verbose: bool = False,
    make_label: bool = True,
) -> dict[str, Any]:
    """

    Args:
        df_series (pl.DataFrame): df_series per sid
        seq_len (int): sequence length
        shift_size (int): shift size
        offset_size (int): offset size
        num_cols (list[str]): numerical columns
        target_cols (list[str]): target columns
        normalize_type (str, optional): normalize type. Defaults to "robust".

    Ref:
    [1]https://github.com/TakoiHirokazu/Kaggle-Parkinsons-Freezing-of-Gait-Prediction/blob/main/takoi/fe/fe022_tdcsfog_1000.ipynb
    """
    num_cols = [
        "anglez",
        "anglez_lag_diff",
        "anglez_lead_diff",
        "anglez_cumsum",
        "enmo",
        "enmo_lag_diff",
        "enmo_lead_diff",
        "enmo_cumsum",
    ]
    target_cols = ["event"]

    # Feature engineering
    cols = ["anglez", "enmo"]
    for col in cols:
        df_series = df_series.with_columns(
            [
                pl.col(f"{col}").diff().alias(f"{col}_lag_diff"),
                pl.col(f"{col}").diff(n=-1).alias(f"{col}_lead_diff"),
                pl.col(f"{col}").cumsum().alias(f"{col}_cumsum"),
            ]
        )

    if verbose:
        print(df_series.shape)
        print(df_series.head(5))
        logger.info(df_series)
    # Cast Target to int
    if make_label:
        label_map = {None: 0, "onset": 1, "wakeup": 2}
        df_series = df_series.with_columns(
            [pl.col("event").map_dict(label_map).cast(pl.Int8).alias("event")]
        )

    # Normalize
    normalize_fn = {
        "mean_std": mean_std_normalize,
        "robust": robust_normalize,
        "standard": standard_normalize,
    }[normalize_type]
    normalized = normalize_fn(
        df_series.select([pl.col(c) for c in num_cols]).to_numpy()
    )
    df_series = df_series.with_columns(
        [pl.Series(normalized[:, i]).alias(num_cols[i]) for i in range(len(num_cols))]
    )
    df_series[num_cols] = df_series[num_cols].fill_nan(0)

    # Make sequence chunks
    batch_size = (df_series.shape[0] - 1) // shift_size
    num_feats = df_series[num_cols].to_numpy()
    if make_label:
        targets = df_series[target_cols].to_numpy()
    else:
        targets = np.zeros((df_series.shape[0], 1), dtype=np.float32)

    time_steps = df_series["step"].to_numpy()

    num_array_ = np.zeros((batch_size, seq_len, len(num_cols)), dtype=np.float32)
    target_array_ = np.zeros((batch_size, seq_len, len(target_cols)), dtype=np.float32)
    mask_array_ = np.zeros((batch_size, seq_len), dtype=np.int8)
    pred_use_array_ = np.zeros((batch_size, seq_len), dtype=np.int8)
    time_array = np.zeros((batch_size, seq_len), dtype=np.int64)
    for _, batch_id in enumerate(range(batch_size)):
        if batch_id == 0:  # first batch
            num_ = num_feats[0:seq_len]
            num_array_[batch_id, :, :] = num_
            target_ = targets[0:seq_len]
            target_array_[batch_id, :, :] = target_
            mask_array_[batch_id, :] = 1
            # : offset_sizeまではb==0のバッチしか使われない, offset_size+shift_size移行は予測に使わない
            pred_use_array_[batch_id, : offset_size + shift_size] = 1
            pred_use_array_[batch_id, 0] = 0
            time_ = time_steps[0:seq_len]
            time_array[batch_id, :] = time_
        elif batch_id == batch_size - 1:  # last batch
            num_ = num_feats[batch_id * shift_size :]
            num_array_[batch_id, : len(num_), :] = num_
            target_ = targets[batch_id * shift_size :]
            target_array_[batch_id, : len(target_), :] = target_
            mask_array_[batch_id, : len(target_)] = 1
            pred_use_array_[batch_id, offset_size : len(target_)] = 1
            time_ = time_steps[batch_id * shift_size :]
            time_array[batch_id, : len(time_)] = time_
        else:
            num_ = num_feats[batch_id * shift_size : (batch_id * shift_size) + seq_len]
            num_array_[batch_id, :, :] = num_
            target_ = targets[batch_id * shift_size : (batch_id * shift_size) + seq_len]
            target_array_[batch_id, :, :] = target_
            mask_array_[batch_id, :] = 1
            pred_use_array_[batch_id, offset_size : offset_size + seq_len] = 1
            time_ = time_steps[
                batch_id * shift_size : (batch_id * shift_size) + seq_len
            ]
            time_array[batch_id, :] = time_

    return {
        "batch_size": batch_size,
        "num_array": num_array_,
        "target_array": target_array_,
        "mask_array": mask_array_,
        "pred_use_array": pred_use_array_,
        "time_array": time_array,
        "ids": [df_series["series_id"][0] for _ in range(batch_size)],
    }


def test_make_seq_chunks():
    from src.utils import LoggingUtils, timer

    logger = LoggingUtils.get_stream_logger(20)

    config = {
        "train_series_path": "./input/child-mind-institute-detect-sleep-states/train_series.parquet",
        "train_events_path": "./input/child-mind-institute-detect-sleep-states/train_events.csv",
    }

    df_series = pl.read_parquet(config["train_series_path"])
    all_ln = df_series.shape[0]
    print("All series shape:", df_series.shape)
    df_events = pl.read_csv(config["train_events_path"])
    df_series = df_series.filter(pl.col("series_id") == "fe90110788d2")
    df_series = df_series.cast({"step": pl.Int64})
    df_events = df_events.filter(pl.col("series_id") == "fe90110788d2")
    df_events = df_events.cast({"step": pl.Int64})
    df_series = df_series.join(df_events, on=["series_id", "step"], how="left")

    print(df_series.shape)
    print("rate -> ", df_series.shape[0] / all_ln)

    with timer("make_sequence_chunks", print):
        seq_chunks = make_sequence_chunks(
            df_series, seq_len=1000, shift_size=500, offset_size=250
        )
    print(seq_chunks["num_array"].shape)
    print(seq_chunks["target_array"].shape)
    print(seq_chunks["mask_array"].shape)
    print(seq_chunks["pred_use_array"].shape)
    print(seq_chunks["time_array"].shape)

    assert seq_chunks["num_array"].shape[1:] == (1000, 8)
    assert seq_chunks["target_array"].shape[1:] == (1000, 1)
    assert seq_chunks["mask_array"].shape[1] == 1000
    assert seq_chunks["pred_use_array"].shape[1] == 1000
    assert seq_chunks["time_array"].shape[1] == 1000

    print(seq_chunks["mask_array"][0])
    print(seq_chunks["time_array"][0])

    print(seq_chunks["mask_array"][-1])
    print(seq_chunks["time_array"][-1])
    # all_ln => 127_946_340
    # 592380/all_ln = 0.0046 = 0.46%
    # 592380 -> 0.296sec = 0.3sec
    # 0.3 * 200 = 60sec = 1min
    # 全体で60secくらいで終わりそう


if __name__ == "__main__":
    test_make_seq_chunks()
