import pathlib
import shutil
from typing import Sequence

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src import utils

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}

FEATURE_NAMES = [
    "anglez",
    "enmo",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()
    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def add_features(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
    ).select("series_id", *FEATURE_NAMES)
    return series_df


def save_each_series(
    this_series_df: pl.DataFrame, columns: Sequence[str], output_dir: pathlib.Path
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    for col in columns:
        x = this_series_df.get_column(col).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col}.npy", x)


class CFG:
    phase: str = "train"
    input_dir: pathlib.Path = pathlib.Path("./input")
    data_dir: pathlib.Path = pathlib.Path(
        "./input/child-mind-institute-detect-sleep-states"
    )


def main(phase: str) -> None:
    cfg = CFG()
    cfg.phase = phase
    if phase == "test":
        cfg.input_dir = pathlib.Path("../input")
        cfg.data_dir = cfg.input_dir / "child-mind-institute-detect-sleep-states"

    processed_dir: pathlib.Path = cfg.input_dir / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {processed_dir}")

    with utils.trace("Load series"):
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                cfg.data_dir / f"{cfg.phase}_series.parquet", low_memory=True
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                processed_dir / "train_series.parquet", low_memory=True
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # Preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with utils.trace("Save Features"):
        for series_id, this_series_df in tqdm(
            series_df.group_by("series_id"), total=n_unique
        ):
            this_series_df = add_features(this_series_df)
            save_each_series(
                this_series_df, FEATURE_NAMES, processed_dir / str(series_id)
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="train")
    args = parser.parse_args()
    main(args.phase)
