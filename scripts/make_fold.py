from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold

from src import utils


class Config:
    n_splits: int = 5
    # n_splits: int = 10
    seed: int = 42

    root_dir: Path = Path(__file__).resolve().parents[1]
    input_dir: Path = root_dir / "input"
    data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
    output_dir: Path = input_dir / "for_train"
    output_dir.mkdir(exist_ok=True, parents=True)
    train_series_path: str | Path = data_dir / "train_series.parquet"
    train_events_path: str | Path = data_dir / "train_events.csv"
    test_series_path: str | Path = data_dir / "test_series.parquet"

    # save_fname: str = "train_series_fold.parquet"
    save_fname: str = f"train_series_fold{n_splits}.parquet"


def make_fold(cfg, n_splits=5) -> pd.DataFrame:
    df_series = pl.read_parquet(cfg.train_series_path)
    df = df_series.to_pandas(use_pyarrow_extension_array=True)
    print(df)

    df["fold"] = -1
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, valid_idx) in enumerate(gkf.split(df, groups=df["series_id"].values)):
        df.loc[valid_idx, "fold"] = fold
    df["fold"] = df["fold"].astype("int")
    return df


if __name__ == "__main__":
    import json

    import numpy as np

    print("root: ", Config.root_dir)
    print("seed: ", Config.seed)
    utils.seed_everything(Config.seed)
    df = make_fold(Config, Config.n_splits)
    print(df)
    print(df.groupby("fold").size())
    print(df.groupby("fold")["series_id"].nunique())
    df.to_parquet(Config.output_dir / Config.save_fname, index=False)

    folded_serieses = []
    for fold in range(Config.n_splits):
        train_series = list(set(df[df["fold"] != fold]["series_id"].to_list()))
        valid_series = list(set(df[df["fold"] == fold]["series_id"].to_list()))
        print(f"fold: {fold}")
        folded_serieses.append(
            {"fold": fold, "train_series": train_series, "valid_series": valid_series}
        )

    with (
        Config.output_dir
        / f"folded_series_ids_fold{Config.n_splits}_seed{Config.seed}.json"
    ).open("w") as fp:
        json.dump(folded_serieses, fp, indent=4)
