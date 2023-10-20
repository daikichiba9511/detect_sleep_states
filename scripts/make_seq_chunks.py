import importlib
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from src.fe import make_sequence_chunks
from src.utils import LoggingUtils, timer

logger = LoggingUtils.get_stream_logger(20)


class FEConfig(Protocol):
    train_series_path: Path
    train_events_path: Path

    normalize_type: str
    seq_len: int
    offset_size: int
    shift_size: int

    save_dir: Path
    num_array_save_fname: str
    target_array_save_fname: str
    mask_array_save_fname: str
    pred_use_array_save_fname: str
    series_ids_array_save_fname: str
    time_array_save_fname: str


def main(config: FEConfig) -> None:
    df_series = pl.read_parquet(config.train_series_path)
    df_events = pl.read_csv(config.train_events_path)
    df_series = df_series.cast({"step": pl.Int64})
    df_events = df_events.cast({"step": pl.Int64})
    df_series = df_series.join(df_events, on=["series_id", "step"], how="left")

    print(
        "Seq_len: ",
        config.seq_len,
        "\nShift_size: ",
        config.shift_size,
        "\nOffset_size: ",
        config.offset_size,
    )

    series_ids = df_series["series_id"].unique().to_list()
    num_array = []
    target_array = []
    mask_array = []
    pred_use_array = []
    series_ids_array = []
    time_array = []
    for series_id in series_ids:
        df_series_sid = df_series.filter(pl.col("series_id") == series_id)
        with timer(f"make_sequence_chunks: {series_id}"):
            seq_chunks = make_sequence_chunks(
                df_series_sid,
                seq_len=config.seq_len,
                shift_size=config.shift_size,
                offset_size=config.offset_size,
                normalize_type=config.normalize_type,
                verbose=True,
            )

        series_ids_array.append([series_id] * seq_chunks["batch_size"])
        num_array.append(seq_chunks["num_array"])
        target_array.append(seq_chunks["target_array"])
        mask_array.append(seq_chunks["mask_array"])
        pred_use_array.append(seq_chunks["pred_use_array"])
        time_array.append(seq_chunks["time_array"])

    num_array = np.concatenate(num_array, axis=0)
    target_array = np.concatenate(target_array, axis=0)
    mask_array = np.concatenate(mask_array, axis=0)
    pred_use_array = np.concatenate(pred_use_array, axis=0)
    series_ids_array = np.concatenate(series_ids_array, axis=0)
    time_array = np.concatenate(time_array, axis=0)

    np.save(config.save_dir / config.num_array_save_fname, num_array)
    np.save(config.save_dir / config.target_array_save_fname, target_array)
    np.save(config.save_dir / config.mask_array_save_fname, mask_array)
    np.save(config.save_dir / config.pred_use_array_save_fname, pred_use_array)
    np.save(config.save_dir / config.series_ids_array_save_fname, series_ids_array)
    np.save(config.save_dir / config.time_array_save_fname, time_array)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default")
    args = parser.parse_args()
    cfg = importlib.import_module(f"src.configs.{args.config}").Config
    main(config=cfg)
