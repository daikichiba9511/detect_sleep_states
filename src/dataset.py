from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.utils import seed_everything


class SleepDataset(Dataset):
    """
    Ref:
        https://www.kaggle.com/code/werus23/child-sleep-critical-point-regression
    """

    def __init__(
        self,
        phase: str,
        series_ids,
        series: pd.DataFrame,
        window_size: int = 12,
    ):
        self.phase = phase
        self.window_size = window_size
        self.series = series
        self.series_ids = series_ids
        self.data = []

        for viz_id in tqdm(self.series_ids, desc="Loading data"):
            self.data.append(
                series.loc[(series.series_id == viz_id)].copy().reset_index(drop=True)
            )

    def downsample_seq_generate_features(self, feat, window_size: int):
        if len(feat) % self.window_size != 0:
            feat = np.concatenate(
                [
                    feat,
                    np.zeros(self.window_size - ((len(feat)) % self.window_size))
                    + feat[-1],
                ]
            )
        # feat: (ori_length, 1) -> (ori_length // ws, ws)
        feat = np.reshape(feat, (-1, window_size))
        feat_mean = np.mean(feat, 1)
        feat_std = np.std(feat, 1)
        feat_median = np.median(feat, 1)
        feat_max = np.max(feat, 1)
        feat_min = np.min(feat, 1)

        return np.dstack([feat_mean, feat_std, feat_median, feat_max, feat_min])[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]

        X = row[["anglez", "enmo"]].to_numpy().astype(np.float32)
        X = np.concatenate(
            [
                self.downsample_seq_generate_features(
                    X[:, i], window_size=self.window_size
                )
                for i in range(X.shape[1])
            ],
            -1,
        )
        X = torch.from_numpy(X)
        if self.phase == "test":
            return X, -999

        y = row[["event"]].to_numpy().astype(np.float32).reshape(-1)
        if len(y) % self.window_size != 0:
            y = np.concatenate(
                [y, np.zeros(self.window_size - ((len(y)) % self.window_size)) + y[-1]]
            )
        y = y.reshape(-1, self.window_size)
        # NOTE: window_size内に1,2がおきると変になるかも
        y = y.max(1)
        y = np.eye(3)[y.astype(np.int32)][:, 1:]
        y = torch.from_numpy(y)
        return X, y


class DataloaderConfig(Protocol):
    seed: int

    train_series_path: str | Path
    train_events_path: str | Path
    test_series_path: str | Path

    batch_size: int
    num_workers: int

    window_size: int


def build_dataloader(
    config: DataloaderConfig, fold: int, phase: str, debug: bool
) -> DataLoader:
    if phase not in ["train", "valid", "test"]:
        raise NotImplementedError

    def collate_fn(data):
        return tuple(zip(*data))

    num_workers = 2 if debug else config.num_workers
    if phase == "test":
        test_series = pl.read_parquet(config.test_series_path)
        ds_test = SleepDataset(
            "test",
            test_series["series_id"].unique(),
            test_series.to_pandas(use_pyarrow_extension_array=True),
            window_size=config.window_size,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        return dl_test

    elif phase == "valid":
        valid_series = pl.read_parquet(config.train_series_path)
        if debug:
            valid_series = valid_series.sample(n=10000)
        valid_series = valid_series.filter(pl.col("fold") == fold)
        valid_series = valid_series.cast({"step": pl.Int64})
        valid_events = pl.read_csv(config.train_events_path, dtypes={"step": pl.Int64})
        df_valid = valid_series.join(valid_events, on=["series_id", "step"], how="left")
        df_valid = df_valid.with_columns(
            pl.col("event").map_dict({"onset": 1, "wakeup": 2}).fill_null(0)
        )
        ds_valid = SleepDataset(
            "valid",
            df_valid["series_id"].unique(),
            df_valid.to_pandas(use_pyarrow_extension_array=True),
            window_size=config.window_size,
        )
        dl_valid = DataLoader(
            ds_valid,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        return dl_valid

    else:
        train_series = pl.read_parquet(config.train_series_path)

        if debug:
            train_series = train_series.sample(n=10000)
        train_series = train_series.filter(pl.col("fold") != fold)
        train_series = train_series.cast({"step": pl.Int64})
        train_events = pl.read_csv(config.train_events_path, dtypes={"step": pl.Int64})
        df_train = train_series.join(train_events, on=["series_id", "step"], how="left")
        df_train = df_train.with_columns(
            pl.col("event").map_dict({"onset": 1, "wakeup": 2}).fill_null(0)
        )
        print(df_train)
        ds_train = SleepDataset(
            "train",
            df_train["series_id"].unique(),
            df_train.to_pandas(use_pyarrow_extension_array=True),
            window_size=config.window_size,
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda _: seed_everything(config.seed),
            collate_fn=collate_fn,
        )
        return dl_train


def test_ds():
    import collections

    config = collections.namedtuple(
        "Config", ["test_series_path", "train_series_path", "train_events_path"]
    )(
        test_series_path="./input/child-mind-institute-detect-sleep-states/test_series.parquet",
        train_series_path="./input/child-mind-institute-detect-sleep-states/train_series.parquet",
        train_events_path="./input/child-mind-institute-detect-sleep-states/train_events.csv",
    )

    test_series = pl.read_parquet(config.test_series_path)
    ds = SleepDataset(
        "test",
        test_series["series_id"].unique(),
        test_series.to_pandas(use_pyarrow_extension_array=True),
    )
    batch = ds[0]
    print(batch[0].shape)


def test_ds2():
    import collections

    config = collections.namedtuple(
        "Config", ["test_series_path", "train_series_path", "train_events_path"]
    )(
        test_series_path="./input/child-mind-institute-detect-sleep-states/test_series.parquet",
        train_series_path="./input/child-mind-institute-detect-sleep-states/train_series.parquet",
        train_events_path="./input/child-mind-institute-detect-sleep-states/train_events.csv",
    )
    train_srs = pl.read_parquet(config.train_series_path, n_rows=50000)
    train_srs = train_srs.cast({"step": pl.Int64})
    print(train_srs)
    train_evs = pl.read_csv(config.train_events_path, dtypes={"step": pl.Int64})
    print(train_evs)

    label_map = {"onset": 1, "wakeup": 2}
    df_train = train_srs.join(train_evs, on=["series_id", "step"], how="left")
    df_train = df_train.with_columns(pl.col("event").map_dict(label_map).fill_null(0))

    print(df_train)
    print(df_train["event"].value_counts())

    ds = SleepDataset(
        "train",
        df_train["series_id"].unique(),
        df_train.to_pandas(use_pyarrow_extension_array=True),
    )
    batch = ds[0]
    img, label = batch
    assert label != -999

    print(img.shape)
    print(label)
    print(label.shape)
    print(label.max())
    print(label.min())
    print(collections.Counter(label.numpy()))


def test_build_dl():
    import collections

    config = collections.namedtuple(
        "Config",
        [
            "seed",
            "test_series_path",
            "train_series_path",
            "train_events_path",
            "batch_size",
            "num_workers",
            "window_size",
        ],
    )(
        seed=42,
        test_series_path="./input/child-mind-institute-detect-sleep-states/test_series.parquet",
        train_series_path="./input/for_train/train_series_fold.parquet",
        train_events_path="./input/child-mind-institute-detect-sleep-states/train_events.csv",
        batch_size=32,
        num_workers=4,
        window_size=12,
    )
    # dl = build_dataloader(config, 0, "test", False)
    # batch = next(iter(dl))
    # print(batch[0].shape)
    # assert isinstance(dl, DataLoader)

    dl = build_dataloader(config, 0, "train", debug=True)
    print(len(dl))
    batch = next(iter(dl))
    print(type(batch[0]))
    print(len((batch[0])))
    print([b.shape for b in batch[0]])
    print(batch[1])
    print([b.shape for b in batch[1]])
    assert isinstance(dl, DataLoader)
    print("build_dataloader test passed")


if __name__ == "__main__":
    # test_ds()
    # test_ds2()
    test_build_dl()
