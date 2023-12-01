import json
import multiprocessing as mp
import pathlib
import random
from pathlib import Path
from typing import Protocol, Sequence, cast, TypeVar

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src import utils as my_utils
from src.fe import make_sequence_chunks
from logging import getLogger

pl.Config.set_tbl_cols(n=300)


logger = getLogger(__name__)


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
        sample_per_epoch: int | None = None,
    ):
        self.phase = phase
        self.window_size = window_size
        self.series = series.reset_index()
        self.series_ids = series_ids
        self.sample_per_epoch = sample_per_epoch
        self.data = []

        for viz_id in tqdm(self.series_ids, desc="Loading data"):
            self.data.append(
                self.series.loc[(self.series.series_id == viz_id)].copy().reset_index()
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
        return (
            len(self.data) if self.sample_per_epoch is None else self.sample_per_epoch
        )

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
            axis=-1,
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
        y = np.eye(3)[y.astype(np.int32)]
        # y = y[:, 1:]
        y = torch.from_numpy(y)
        return X, y


def cleaning(data: pl.DataFrame) -> pl.DataFrame:
    data = data.drop_nulls(subset=["timestamp"])
    return data


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
        print("######### Valid ###########")
        valid_series = pl.read_parquet(config.train_series_path)
        if debug:
            valid_series = valid_series.sample(n=10000)
        valid_series = valid_series.filter(pl.col("fold") == fold)
        valid_series = valid_series.cast({"step": pl.Int64})
        valid_events = pl.read_csv(config.train_events_path, dtypes={"step": pl.Int64})
        df_valid = valid_series.join(valid_events, on=["series_id", "step"], how="left")
        df_valid = df_valid.with_columns(
            pl.col("event").map_dict({"onset": 1, "wakeup": 2, None: 0})
        )
        print("Before: ", len(df_valid))
        df_valid = cleaning(df_valid)
        print("After: ", len(df_valid))
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
        print("######### Train ###########")
        train_series = pl.read_parquet(config.train_series_path)
        if debug:
            train_series = train_series.sample(n=10000)
        train_series = train_series.filter(pl.col("fold") != fold)
        train_series = train_series.cast({"step": pl.Int64})
        train_events = pl.read_csv(config.train_events_path, dtypes={"step": pl.Int64})
        df_train = train_series.join(train_events, on=["series_id", "step"], how="left")
        df_train = df_train.with_columns(
            pl.col("event").map_dict({"onset": 1, "wakeup": 2, None: 0})
        )
        print("Before: ", len(df_train))
        df_train = cleaning(df_train)
        print("After: ", len(df_train))
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
            worker_init_fn=lambda _: my_utils.seed_everything(config.seed),
            collate_fn=collate_fn,
        )
        return dl_train


###################################################
# SleepDatasetV2
###################################################


def preprcess(num_array: np.ndarray, mask_array: np.ndarray) -> dict[str, np.ndarray]:
    attention_mask = mask_array == 0
    return {
        "input_data_num_array": num_array,
        "input_data_mask_array": mask_array,
        "attention_mask": attention_mask,
    }


class SleepDatasetV2(Dataset):
    """
    Ref:
        https://www.kaggle.com/code/werus23/child-sleep-critical-point-regression
    """

    def __init__(
        self,
        phase: str,
        num_array: np.ndarray,
        mask_array: np.ndarray,
        time_array: np.ndarray,
        pred_use_array: np.ndarray,
        series_ids_array: np.ndarray,
        y: np.ndarray | None = None,
    ):
        if phase not in ["train", "valid", "test"]:
            raise NotImplementedError
        if phase in ["train", "valid"] and y is None:
            raise ValueError("y must be not None when phase is train or valid")

        self.phase = phase
        self.num_array = num_array
        self.mask_array = mask_array
        self.time_array = time_array
        self.pred_use_array = pred_use_array
        self.series_ids_array = series_ids_array
        self.y = y

    def __len__(self):
        return len(self.num_array)

    def __getitem__(self, index) -> dict[str, torch.Tensor | np.ndarray]:
        data = preprcess(self.num_array[index], self.mask_array[index])
        if self.phase == "test":
            return {
                "input_data_num_array": torch.tensor(
                    data["input_data_num_array"], dtype=torch.float32
                ),
                "input_data_mask_array": torch.tensor(
                    data["input_data_mask_array"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    data["attention_mask"], dtype=torch.float32
                ),
                "steps": torch.tensor(self.time_array[index], dtype=torch.long),
                "pred_use_array": torch.tensor(
                    self.pred_use_array[index], dtype=torch.long
                ),
                "series_ids": self.series_ids_array[index],
            }
        assert isinstance(self.y, np.ndarray)
        return {
            "input_data_num_array": torch.tensor(
                data["input_data_num_array"], dtype=torch.float32
            ),
            "input_data_mask_array": torch.tensor(
                data["input_data_mask_array"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(data["attention_mask"], dtype=torch.float32),
            "y": torch.tensor(self.y[index], dtype=torch.float32),
            "steps": torch.tensor(self.time_array[index], dtype=torch.long),
            "pred_use_array": torch.tensor(
                self.pred_use_array[index], dtype=torch.long
            ),
            "series_ids": self.series_ids_array[index],
        }


def filtering(
    num_array: np.ndarray,
    mask_array: np.ndarray,
    pred_use_array: np.ndarray,
    target_array: np.ndarray,
    time_array: np.ndarray,
):
    filtered_num_array = []
    filtered_mask_array = []
    filtered_pred_use_array = []
    filtered_target_array = []
    filtered_time_array = []
    for i in range(len(num_array)):
        num = num_array[i]
        mask = mask_array[i]
        pred_use = pred_use_array[i]
        target = target_array[i]  # (seq_len, 1)
        time = time_array[i]

        # target.sum == 0はonset/wakeupが一つもないことを意味する
        if target.sum() != 0:
            filtered_num_array.append(num)
            filtered_mask_array.append(mask)
            filtered_pred_use_array.append(pred_use)
            filtered_target_array.append(target)
            filtered_time_array.append(time)
    filtered_num_array = np.stack(filtered_num_array, axis=0)
    filtered_mask_array = np.stack(filtered_mask_array, axis=0)
    filtered_pred_use_array = np.stack(filtered_pred_use_array, axis=0)
    filtered_target_array = np.stack(filtered_target_array, axis=0)
    filtered_time_array = np.stack(filtered_time_array, axis=0)
    return (
        filtered_num_array,
        filtered_mask_array,
        filtered_pred_use_array,
        filtered_target_array,
        filtered_time_array,
    )


class DataloaderConfigV2(Protocol):
    seed: int

    train_series_path: str | Path
    train_events_path: str | Path
    test_series_path: str | Path

    mask_array_path: Path
    num_array_path: Path
    pred_use_array_path: Path
    series_ids_array_path: Path
    target_array_path: Path
    time_array_path: Path

    seq_len: int
    shift_size: int
    offset_size: int

    batch_size: int
    num_workers: int

    window_size: int
    out_size: int


def build_dataloader_v2(
    config: DataloaderConfigV2,
    fold: int,
    phase: str,
    debug: bool,
    use_cache: bool = True,
) -> DataLoader:
    if phase not in ["train", "valid", "test"]:
        raise NotImplementedError

    def collate_fn(data):
        batch = tuple(zip(*data))
        return batch

    num_workers = 2 if debug else config.num_workers
    if phase == "test":
        test_series = pl.read_parquet(config.test_series_path)

        data = make_sequence_chunks(
            test_series,
            seq_len=config.seq_len,
            shift_size=config.shift_size,
            offset_size=config.offset_size,
            normalize_type="robust",
            make_label=False,
        )

        ds_test = SleepDatasetV2(
            "test",
            num_array=data["num_array"],
            mask_array=data["mask_array"],
            time_array=data["time_array"],
            pred_use_array=data["pred_use_array"],
            series_ids_array=data["series_ids_array"],
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            # collate_fn=collate_fn,
        )
        return dl_test

    elif phase == "valid":
        print("######### Valid ###########")
        valid_series = pl.read_parquet(config.train_series_path)
        if debug:
            valid_series = valid_series.sample(n=10000)
        valid_series = valid_series.filter(pl.col("fold") == fold)
        valid_series_ids = valid_series["series_id"].unique().to_list()

        # print(valid_series_ids)

        valid_id_mask = np.isin(np.load(config.series_ids_array_path), valid_series_ids)

        if use_cache:
            num_array = np.load(config.num_array_path)[valid_id_mask]
            mask_array = np.load(config.mask_array_path)[valid_id_mask]
            target_array = np.load(config.target_array_path)[valid_id_mask].astype(
                np.int32
            )
            time_array = np.load(config.time_array_path)[valid_id_mask]
            pred_use_array = np.load(config.pred_use_array_path)[valid_id_mask]
            series_ids_array = np.load(config.series_ids_array_path)[valid_id_mask]
        else:
            print("############## Make sequence chunks ##############")
            df_events = pl.read_csv(config.train_events_path)
            df_events = df_events.cast({"step": pl.Int64})
            valid_series = valid_series.cast({"step": pl.Int64})
            valid_series = valid_series.join(
                df_events, on=["series_id", "step"], how="left"
            )

            series_ids = valid_series["series_id"].unique().to_list()
            num_array = []
            target_array = []
            mask_array = []
            pred_use_array = []
            series_ids_array = []
            time_array = []
            for series_id in series_ids:
                df_series_sid = valid_series.filter(pl.col("series_id") == series_id)
                seq_chunks = make_sequence_chunks(
                    df_series_sid,
                    seq_len=config.seq_len,
                    shift_size=config.shift_size,
                    offset_size=config.offset_size,
                    normalize_type="robust",
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

        print(np.unique(series_ids_array))

        target_array = target_array.astype(int)
        y = []
        for i in range(len(target_array)):
            target_ = target_array[i]
            onehot = np.zeros((len(target_), config.out_size))
            onehot[np.arange(len(target_)), target_[:, 0]] = 1
            y.append(onehot)
        y = np.concatenate(y, axis=0).reshape(-1, num_array.shape[1], config.out_size)

        ds_valid = SleepDatasetV2(
            "valid",
            num_array=num_array,
            mask_array=mask_array,
            time_array=time_array,
            pred_use_array=pred_use_array,
            series_ids_array=series_ids_array,
            y=y,
        )
        dl_valid = DataLoader(
            ds_valid,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            # collate_fn=collate_fn,
        )
        return dl_valid

    else:
        # TODO: foldのseries_idを知るのをもっと効率よくする
        print("######### Train ###########")
        train_series = pl.read_parquet(config.train_series_path)
        if debug:
            train_series = train_series.sample(n=10000)
        train_series = train_series.filter(pl.col("fold") != fold)
        train_series_ids = train_series["series_id"].unique().to_list()

        # with Path("./output/series/series_id_null_label.json").open("r") as fp:
        #     series_id_null_label = json.load(fp)
        # not_null_series_ids = series_id_null_label["not_have_null"]
        # assert len(not_null_series_ids) == 37
        # train_series_ids = [
        #     sid for sid in train_series_ids if sid in not_null_series_ids
        # ]
        print(
            "length of train series don't have null step event", len(train_series_ids)
        )

        train_id_mask = np.isin(np.load(config.series_ids_array_path), train_series_ids)

        num_array = np.load(config.num_array_path)[train_id_mask]
        mask_array = np.load(config.mask_array_path)[train_id_mask]
        target_array = np.load(config.target_array_path)[train_id_mask].astype(np.int32)
        time_array = np.load(config.time_array_path)[train_id_mask]
        pred_use_array = np.load(config.pred_use_array_path)[train_id_mask]
        series_ids_array = np.load(config.series_ids_array_path)[train_id_mask]

        print("Filter Before: ", num_array.shape)
        num_array, mask_array, pred_use_array, target_array, time_array = filtering(
            num_array, mask_array, pred_use_array, target_array, time_array
        )
        print("Filter After: ", num_array.shape)

        y = []
        for i in range(len(target_array)):
            target_ = target_array[i]
            onehot = np.zeros((len(target_), config.out_size))
            onehot[np.arange(len(target_)), target_[:, 0]] = 1
            y.append(onehot)
        y = np.concatenate(y, axis=0).reshape(-1, num_array.shape[1], config.out_size)

        ds_train = SleepDatasetV2(
            "train",
            num_array=num_array,
            mask_array=mask_array,
            time_array=time_array,
            pred_use_array=pred_use_array,
            series_ids_array=series_ids_array,
            y=y,
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda _: my_utils.seed_everything(config.seed),
            # collate_fn=collate_fn,
        )
        return dl_train


############################################
# SleepDatasetV3
############################################
def mean_std_normalize_label(y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mean_0 = y[:, 0].mean().item()
    std_0 = y[:, 0].std().item()
    y[:, 0] = (y[:, 0] - mean_0) / (std_0 + eps)

    mean_1 = y[:, 1].mean().item()
    std_1 = y[:, 1].std().item()
    y[:, 1] = (y[:, 1] - mean_1) / (std_1 + eps)
    return y


def normalize(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + eps)
    return x


def min_max_normalize(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return x


class SleepDatasetV3(Dataset):
    def __init__(
        self,
        phase: str,
        data: list[pl.DataFrame],
        downsample_factor: int,
        ids: list[str],
        targets: list[list[tuple[int, int]]] | None,
        sigma: int | None,
        w_sigma: float | None,
        seq_len: int = 1000,
        random_sequence_mixing: bool = False,
        sample_per_epoch: int = 20000,
    ):
        if phase not in ["train", "valid", "test"]:
            raise NotImplementedError

        if phase in ["train", "valid"] and targets is None:
            raise ValueError("targets must be not None when phase is train or valid")

        if phase in ["train", "valid"] and sigma is None:
            raise ValueError("sigma must be not None when phase is train or valid")

        if phase in ["train", "valid"] and w_sigma is None:
            raise ValueError("w_sigma must be not None when phase is train or valid")

        self.phase = phase
        self.targets = targets

        self.data = []
        # Normalize
        normalize_fn = normalize
        for df in tqdm(data, desc="Preprocessing data"):
            for col in ["anglez", "enmo"]:
                df.with_columns(
                    pl.col(col)
                    .map_batches(lambda x: pl.Series(normalize_fn(x.to_numpy())))
                    .alias(col)
                )
            self.data.append(df.to_pandas(use_pyarrow_extension_array=True))

        self.ids = ids
        self.sigma = sigma
        self.downsample_factor = downsample_factor
        self.w_sigma = w_sigma
        self.seq_len = seq_len
        self.sample_per_epoch = sample_per_epoch
        self.random_sequence_mixing = random_sequence_mixing

    def __len__(self) -> int:
        return self.sample_per_epoch if self.phase == "train" else len(self.data)

    def downsample_and_create_feats(self, feat: np.ndarray, downsample_factor: int):
        # downsample
        # 0-padding
        if len(feat) % downsample_factor != 0:
            zeros = (
                np.zeros(downsample_factor - (len(feat) % downsample_factor)) + feat[-1]
            )
            feat = np.concatenate([feat, zeros])

        feat = np.reshape(feat, (-1, downsample_factor)).astype(np.float32)
        feat_mean = np.mean(feat, 1)
        feat_std = np.std(feat, 1)
        feat_median = np.median(feat, 1)
        feat_max = np.max(feat, 1)
        feat_min = np.min(feat, 1)
        # shift_feat_mean = np.roll(feat_mean, 1)
        # shift2_feat_mean = np.roll(feat_mean, 2)
        # diff_featmean_featmean_mean = feat_mean - np.mean(feat_mean)
        # diff_featmean_featmean_median = feat_mean - np.median(feat_mean)
        feat = np.dstack(
            [
                feat_mean,
                feat_std,
                feat_median,
                feat_max,
                feat_min,
                # shift_feat_mean,
                # shift2_feat_mean,
                # diff_featmean_featmean_mean,
                # diff_featmean_featmean_median,
            ]
        )[0]
        return feat

    def downsample_sequence(self, feat, downsample_factor: int):
        # downsample
        # 0-padding
        if len(feat) % downsample_factor != 0:
            zeros = np.zeros(
                downsample_factor - (len(feat) % downsample_factor) + feat[-1]
            )
            feat = np.concatenate([feat, zeros])

        feat = np.reshape(feat, (-1, downsample_factor))
        feat = np.mean(feat, 1)
        return feat

    def gaussian(self, n: int, sigma: float):
        # gaussian distribution function
        r = np.arange(-n // 2, n // 2 + 1)
        return [
            1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(float(x) ** 2) / (2 * sigma**2))
            for x in r
        ]

    def _make_label(self, data_i: np.ndarray, index: int) -> torch.Tensor:
        if self.targets is None or self.sigma is None or self.w_sigma is None:
            raise ValueError(
                "targets, sigma, downsample_factor must be not None when phase is train or valid"
            )

        # yには対応するseriesのevent_range: list[(start, end)]が入ってる
        target_ranges = self.targets[index]
        sigma = self.sigma
        # targetの値をガウス分布に変換
        target_gaussian = np.zeros((len(data_i), 2))
        gause_pdf_v = self.gaussian(n=sigma, sigma=sigma * self.w_sigma)
        for start, end in target_ranges:
            # start, endの点を中心にsigmaの範囲でガウス分布を作成
            s1 = max(0, start - self.sigma // 2)
            s2 = start + self.sigma // 2 + 1
            e1 = end - self.sigma // 2
            e2 = min(len(data_i), end + self.sigma // 2 + 1)

            target_gaussian[s1:s2, 0] = gause_pdf_v[s1 - (start - sigma // 2) :]
            target_gaussian[e1:e2, 1] = gause_pdf_v[
                : sigma + 1 - ((end + sigma // 2 + 1) - e2)
            ]
        y = target_gaussian
        y = np.dstack(
            [
                self.downsample_sequence(y[:, i], self.downsample_factor)
                for i in range(y.shape[-1])
            ]
        )[0]
        y = mean_std_normalize_label(torch.tensor(y).float())
        return y

    def __getitem__(self, index: int):
        if self.phase == "test":
            data_i = self.data[index][["anglez", "enmo"]].to_numpy()
            sid = self.ids[index]
            step = self.data[index]["step"].to_numpy().astype(int)
            # step = step[:: self.downsample_factor]

            X = np.concatenate(
                [
                    self.downsample_and_create_feats(
                        data_i[:, i], self.downsample_factor
                    )
                    for i in range(data_i.shape[-1])
                ],
                axis=-1,
            )
            X = torch.from_numpy(X).float()
            return X, -np.inf, sid, step

        if self.phase == "valid":
            data_i = self.data[index][["anglez", "enmo"]].to_numpy()
            sid = self.ids[index]
            step = self.data[index]["step"].to_numpy().astype(int)
            # step = step[:: self.downsample_factor]

            X = np.concatenate(
                [
                    self.downsample_and_create_feats(
                        data_i[:, i], self.downsample_factor
                    )
                    for i in range(data_i.shape[-1])
                ],
                axis=-1,
            )
            X = torch.from_numpy(X).float()
            y = self._make_label(data_i, index)
            return X, y, sid, step

        # Train
        # バッチ内に同じseries_idが固まらないようにする
        index_ = np.random.randint(0, len(self.data))
        data_i = self.data[index_][["anglez", "enmo"]].to_numpy()
        sid = self.ids[index_]
        step = self.data[index_]["step"].to_numpy().astype(int)
        step = step[:: self.downsample_factor]
        X = np.concatenate(
            [
                self.downsample_and_create_feats(data_i[:, i], self.downsample_factor)
                for i in range(data_i.shape[-1])
            ],
            axis=-1,
        )
        X = torch.tensor(X).float()
        # random sampling
        y = self._make_label(data_i, index_)

        # random sequence mixing
        if self.random_sequence_mixing and np.random.rand() < 0.5:
            # print("########## Random sequence mixing ##########")
            index2_ = np.random.randint(0, len(self.data))
            data2_i = self.data[index2_][["anglez", "enmo"]].to_numpy()
            sid2 = self.ids[index2_]
            step2 = self.data[index2_]["step"].to_numpy().astype(int)
            step2 = step2[:: self.downsample_factor]
            X2 = np.concatenate(
                [
                    self.downsample_and_create_feats(
                        data2_i[:, i], self.downsample_factor
                    )
                    for i in range(data2_i.shape[-1])
                ],
                axis=-1,
            )
            X2 = torch.tensor(X2).float()
            y2 = self._make_label(data2_i, index2_)

            start1 = np.random.randint(low=0, high=max(1, len(X) - self.seq_len))
            end1 = min(start1 + self.seq_len // 2, len(X))
            start2 = np.random.randint(low=0, high=max(1, len(X) - self.seq_len))
            end2 = min(start2 + self.seq_len // 2, len(X))

            X = torch.cat([X[start1:end1], X2[start2:end2]], dim=0)
            y = torch.cat([y[start1:end1], y2[start2:end2]], dim=0)
            step = np.concatenate([step[start1:end1], step2[start2:end2]], axis=0)
            if len(X) < self.seq_len:
                X = torch.cat(
                    [X, torch.zeros(self.seq_len - len(X), X.shape[-1])], dim=0
                )
            if len(y) < self.seq_len:
                y = torch.cat(
                    [y, torch.zeros(self.seq_len - len(y), y.shape[-1])], dim=0
                )
            if len(step) < self.seq_len:
                step = np.concatenate(
                    [step, np.zeros(self.seq_len - len(step), dtype=int)], axis=0
                )
            # sidはtrainのときは使わないので今はとりあえず考えない
            return X, y, sid, step

        # Normal random sequence sampling
        start = np.random.randint(low=0, high=max(1, len(X) - self.seq_len))
        end = min(start + self.seq_len, len(X))
        X = X[start:end]
        if len(X) < self.seq_len:
            X = torch.cat([X, torch.zeros(self.seq_len - len(X), X.shape[-1])], dim=0)
        y = y[start:end]
        if len(y) < self.seq_len:
            y = torch.cat([y, torch.zeros(self.seq_len - len(y), y.shape[-1])], dim=0)

        step = step[start:end]
        if len(step) < self.seq_len:
            step = np.concatenate(
                [step, np.zeros(self.seq_len - len(step), dtype=int)], axis=0
            )

        return X, y, sid, step


class DataloaderConfigV3(Protocol):
    seed: int

    train_series_path: str | Path
    train_events_path: str | Path
    test_series_path: str | Path

    batch_size: int
    num_workers: int

    window_size: int
    out_size: int

    sigma: int
    """default: 720"""
    downsample_factor: int
    """default: 12 <=> 1 sample/min"""
    w_sigma: float
    """default: 0.15"""

    series_dir: Path
    target_series_uni_ids_path: Path

    train_seq_len: int
    # infer_seq_len: int

    random_sequence_mixing: bool
    """ランダムに２つシーケンスを選んで、seq_lenの半分づつくっつける. 正則化を期待"""

    sample_per_epoch: int
    """trainのときにepochあたり何サンプル使うか"""


def build_dataloader_v3(
    config: DataloaderConfigV3,
    fold: int,
    phase: str,
    debug: bool,
) -> DataLoader:
    if phase not in ["train", "valid", "test"]:
        raise NotImplementedError

    num_workers = 2 if debug else config.num_workers
    if phase == "test":
        print("######### Test ###########")
        df_test_series = pl.read_parquet(config.test_series_path)
        data_test_series = []
        sid_test = []
        for sid in df_test_series["series_id"].unique():
            data_test_series.append(
                df_test_series.filter(pl.col("series_id") == sid).to_pandas(
                    use_pyarrow_extension_array=True
                )
            )
            sid_test.append(sid)
        ds_test = SleepDatasetV3(
            phase,
            targets=None,
            data=data_test_series,
            ids=sid_test,
            sigma=config.sigma,
            downsample_factor=config.downsample_factor,
            w_sigma=config.w_sigma,
            # seq_len=config.infer_seq_len,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            # collate_fn=collate_fn,
            persistent_workers=True,  # data loaderが使い終わったProcessをkillしない
        )
        return dl_test
    elif phase == "valid":
        print("######### Valid ###########")
        df_valid_series = pl.read_parquet(config.train_series_path)
        df_valid_series = df_valid_series.filter(pl.col("fold") == fold)
        valid_series_ids = df_valid_series["series_id"].unique().to_list()

        targets, data_valid_series, ids = joblib.load(config.target_series_uni_ids_path)

        valid_targets: list[list[tuple[int, int]]] = []
        valid_data: list[pl.DataFrame] = []
        valid_ids: list[str] = []
        for i, sid in enumerate(ids):
            if sid in valid_series_ids:
                valid_targets.append(targets[i])
                valid_data.append(data_valid_series[i])
                valid_ids.append(sid)
            if debug and i > 50:
                break

        ds_valid = SleepDatasetV3(
            phase,
            targets=valid_targets,
            data=valid_data,
            ids=valid_ids,
            sigma=config.sigma,
            downsample_factor=config.downsample_factor,
            w_sigma=config.w_sigma,
            # seq_len=config.infer_seq_len,
        )
        dl_valid = DataLoader(
            ds_valid,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            # collate_fn=collate_fn,
            persistent_workers=True,  # data loaderが使い終わったProcessをkillしない
        )
        return dl_valid
    else:
        print("######### Train ###########")
        df_train_series = pl.read_parquet(config.train_series_path)
        df_train_series = df_train_series.filter(pl.col("fold") != fold)
        train_series_ids = df_train_series["series_id"].unique().to_list()
        print(len(train_series_ids))

        targets, data_train_series, ids = joblib.load(config.target_series_uni_ids_path)

        train_targets: list[list[tuple[int, int]]] = []
        train_data: list[pl.DataFrame] = []
        train_ids: list[str] = []
        for i, sid in enumerate(ids):
            if sid in train_series_ids:
                train_targets.append(targets[i])
                train_data.append(data_train_series[i])
                train_ids.append(sid)
            if debug and i > 50:
                break

        print("Train data size: ", len(train_data))

        ds_train = SleepDatasetV3(
            phase,
            targets=train_targets,
            data=train_data,
            ids=train_ids,
            sigma=config.sigma,
            downsample_factor=config.downsample_factor,
            w_sigma=config.w_sigma,
            seq_len=config.train_seq_len,
            random_sequence_mixing=config.random_sequence_mixing,
            sample_per_epoch=config.sample_per_epoch,
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda _: my_utils.seed_everything(config.seed),
            # collate_fn=collate_fn,
            persistent_workers=True,  # data loaderが使い終わったProcessをkillしない
        )
        return dl_train


############################################
# SleepSegDataset
############################################
def load_features(
    feature_names: Sequence[str],
    series_ids: Sequence[str] | None,
    processed_dir: Path,
    phase: str,
    do_min_max_normalize: bool,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [
            series_dir.name for series_dir in (processed_dir / phase).glob("*")
        ]

    for series_id in tqdm(series_ids, desc="Load features"):
        series_dir = processed_dir / series_id
        this_features = []
        for feature_name in feature_names:
            feature = np.load(series_dir / f"{feature_name}.npy").astype(np.float32)
            if do_min_max_normalize and feature_name in [
                "anglez",
                "enmo",
                "hour_sin",
                "hour_cos",
            ]:
                feature = min_max_normalize(feature, eps=1e-7)
            this_features.append(feature)
        features[series_id] = np.stack(this_features, axis=1)
    return features


def load_chunk_features(
    seq_len: int,
    feature_names: Sequence[str],
    series_ids: Sequence[str] | None,
    processed_dir: Path,
    phase: str,
    slide_size: int,
    do_min_max_normalize: bool,
) -> dict[str, np.ndarray]:
    if series_ids is None:
        series_ids = [
            series_dir.name for series_dir in (processed_dir / phase).glob("*")
        ]

    features = {}
    for series_id in tqdm(series_ids, desc="Load features"):
        series_dir = processed_dir / series_id

        # (steps, n_features)
        this_features = []
        for feature_name in feature_names:
            feature = np.load(series_dir / f"{feature_name}.npy").astype(np.float32)
            if do_min_max_normalize and feature_name in [
                "anglez",
                "enmo",
                "hour_sin",
                "hour_cos",
            ]:
                feature = min_max_normalize(feature, eps=1e-7)
            this_features.append(feature)
        this_features = np.stack(this_features, axis=1)

        num_chunks = (len(this_features) // slide_size) + 1
        for i in range(num_chunks):
            start = i * slide_size
            end = start + seq_len
            chunk_feats = my_utils.pad_if_needed(
                this_features[start:end], seq_len, pad_value=0
            )
            features[f"{series_id}_{i:07}"] = chunk_feats
    return features


def random_crop(pos: int, duration: int, max_end: int) -> tuple[int, int]:
    # 0 <= start <= pos - duration
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


def make_label(
    this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int
) -> np.ndarray:
    def _make_pos_on_num_frames(pos: int, num_frames: int, start: int) -> int:
        relatiev_pos = (pos - start) / duration
        return int(relatiev_pos * num_frames)

    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, 3))
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        # make relative position in the num_frames
        onset_pos_on_num_frames = _make_pos_on_num_frames(onset, num_frames, start)
        if 0 <= onset_pos_on_num_frames < num_frames:
            label[onset_pos_on_num_frames, 1] = 1

        wakeup_pos_on_num_frames = _make_pos_on_num_frames(wakeup, num_frames, start)
        if 0 <= wakeup_pos_on_num_frames < num_frames:
            label[wakeup_pos_on_num_frames, 2] = 1

        # make sleep label (class 0)
        onset_pos_on_num_frames = max(0, onset_pos_on_num_frames)
        wakeup_pos_on_num_frames = min(num_frames, wakeup_pos_on_num_frames)
        label[onset_pos_on_num_frames:wakeup_pos_on_num_frames, 0] = 1

    return label


def gaussian_kernel(length: int, sigma: float) -> np.ndarray:
    x = cast(np.ndarray, np.ogrid[-length : length + 1])
    g = np.exp(-(x**2 / (2 * sigma**2)))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    """
    Args:
        label: (num_frames, n_classes)
    """
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(
            label[:, i], gaussian_kernel(offset, sigma), mode="same"
        )
    return label


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """num_stepsの中から、this_event_dfのonset, wakeupとかぶらないようにランダムにサンプリングする

    Args:
        this_event_df: (num_events, 2)
        num_steps: int

    Returns:
        int: negative position
    """
    pos_positions = set(
        this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist()
    )
    neg_positions = list(set(range(num_steps)) - pos_positions)
    return random.sample(neg_positions, 1)[0]


_T = TypeVar("_T", np.ndarray, torch.Tensor)


def label_smoothing(label: _T, eps: float = 0.1) -> _T:
    """
    Args:
        label: (num_frames, n_classes)
    """
    num_classes = label.shape[1]
    # label smoothing for sleep(class 0)
    label[0] = (1 - eps) * label[0] + eps / num_classes
    return label


def nearest_valid_size(input_size: int, downsample_factor: int) -> int:
    while (input_size // downsample_factor) % 32 != 0:
        input_size += 1
    return input_size


def make_periodic_mask(
    periodic_steps: np.ndarray, start: int, end: int, seq_len: int, label_length: int
) -> np.ndarray:
    """周期的なステップを指すマスクを作成する

    Args:
        periodic_steps: (periodic_steps, )
        start:
        end:
        seq_len: int
        label_length: int
    """
    mask_step = [
        label_length * int((step - start) // seq_len)
        for step in periodic_steps
        if start <= step <= end
    ]
    label_mask = np.zeros(label_length)
    label_mask[mask_step] = 1
    return mask


class TrainSegDatasetConfig(Protocol):
    data_dir: pathlib.Path
    processed_dir: pathlib.Path
    seed: int

    train_series: list[str]

    features: list[str]

    seq_len: int
    """系列長の長さ"""
    upsample_rate: float
    downsample_rate: int
    bg_sampling_rate: float
    offset: int
    sigma: int


class SleepSegTrainDataset(Dataset):
    def __init__(
        self,
        cfg: TrainSegDatasetConfig,
        df: pl.DataFrame,
        features: dict[str, np.ndarray],
        sample_per_epoch: int | None,
        train_periodic_dict: dict[str, np.ndarray] | None = None,
        do_sleep_label_smoothing: bool = False,
    ) -> None:
        self.cfg = cfg

        self.event_df = (
            # TODO: nightに重複があってここでエラーでるので、aggregate_function="min"で対応してる<=>完全に活用できてるわけじゃない
            df.pivot(
                index=["series_id", "night"],
                columns="event",
                values="step",
                aggregate_function="min",
            )
            .drop_nulls()
            .to_pandas(use_pyarrow_extension_array=True)
        )

        # the form of event_df is following
        #
        # | series_id | night | onset | wakeup |
        # |-----------|-------|-------|--------|
        # | 03d55.... | 1     | 1000  | 5000   |
        # | ...       | ...   | ...   | ...    |
        #
        self.seires_event_df_map = {
            series_id: series_df
            for series_id, series_df in self.event_df.groupby("series_id")
        }

        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(cfg.seq_len * cfg.upsample_rate), cfg.downsample_rate
        )
        self.sample_per_epoch = sample_per_epoch
        self.bg_sampling_rate = cfg.bg_sampling_rate
        self.do_sleep_label_smoothing = do_sleep_label_smoothing

        self.train_periodic_dict = train_periodic_dict

        # exp053
        series_weights_path = pathlib.Path(
            "./output/series_weights/series_weights.json"
        )
        if series_weights_path.exists():
            with series_weights_path.open("r") as f:
                self.series_weights = json.load(f)
                self.series_weights = {
                    d["series_id"]: d["weight"] for d in self.series_weights
                }

    def __len__(self) -> int:
        return (
            len(self.event_df)
            if self.sample_per_epoch is None
            else self.sample_per_epoch
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])  # type: ignore
        # the form of event_df is following
        #
        # | series_id | night | onset | wakeup |
        # |-----------|-------|-------|--------|
        # | 03d55.... | 1     | 1000  | 5000   |
        # | ...       | ...   | ...   | ...    |
        #
        event_df = self.event_df
        step = event_df.at[index, event]
        if pd.isna(step):
            other_event = "wakeup" if event == "onset" else "onset"
            step = event_df.at[index, other_event]

        series_id = event_df.at[index, "series_id"]

        # series_df = event_df.query("series_id == @series_id").reset_index(drop=True)
        series_df = self.seires_event_df_map[series_id].reset_index(drop=True)

        this_series_features = self.features[series_id]
        n_steps = len(this_series_features)

        series_weight = self.series_weights.get(series_id, 1.0)

        if random.random() < self.bg_sampling_rate:
            step = negative_sampling(series_df, n_steps)

        # crop
        start, end = random_crop(step, self.cfg.seq_len, n_steps)
        feature = this_series_features[start:end]  # (seq_len, num_features)

        # upsample seq_len to upsampled_num_frames
        # (1, num_features, seq_len)
        feature = torch.FloatTensor(feature.T).unsqueeze(0)
        feature = TF.resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        )
        feature = feature.squeeze(0)

        # hard label to gaussian label
        # lengthをdownsamplingする
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = make_label(series_df, num_frames, self.cfg.seq_len, start, end)
        label = label.astype(np.float32)
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]], self.cfg.offset, self.cfg.sigma
        )
        if self.do_sleep_label_smoothing:
            label = label_smoothing(label, eps=0.1)

        if self.train_periodic_dict is not None:
            periodic = self.train_periodic_dict[series_id]
            periodic_mask = make_periodic_mask(
                periodic, start, end, self.cfg.seq_len, label.shape[0]
            )
            periodic_mask = torch.FloatTensor(periodic_mask)
        else:
            periodic_mask = torch.zeros(label.shape[0])

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (num_frames, 3) (2880, 3)
            "weight": series_weight,
            "periodic_mask": periodic_mask,
        }


class ValidSegDatasetConfig(Protocol):
    data_dir: pathlib.Path
    processed_dir: pathlib.Path
    seed: int

    valid_series: list[str]

    seq_len: int
    """系列長の長さ"""
    upsample_rate: float
    downsample_rate: int
    features: list[str]


class SleepSegValidDataset(Dataset):
    def __init__(
        self,
        cfg: ValidSegDatasetConfig,
        chunk_features: dict[str, np.ndarray],
        df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            df.pivot(
                index=["series_id", "night"],
                columns="event",
                values="step",
                aggregate_function="min",
            )
            .drop_nulls()
            .to_pandas(use_pyarrow_extension_array=True)
        )
        self.num_features = len(chunk_features)
        self.upsampled_num_frames = nearest_valid_size(
            int(cfg.seq_len * cfg.upsample_rate), cfg.downsample_rate
        )
        self.num_features = len(cfg.features)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]
        feature = self.chunk_features[key]
        # (1, num_features, seq_len)
        feature = torch.FloatTensor(feature.T).unsqueeze(0)
        feature = TF.resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        )
        feature = feature.squeeze(0)
        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.seq_len
        end = start + self.cfg.seq_len
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = make_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.seq_len,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (seq_len, 3)
        }


class TestSegDatasetConfig(Protocol):
    data_dir: pathlib.Path
    processed_dir: pathlib.Path
    seed: int

    seq_len: int
    """系列長の長さ"""
    upsample_rate: float
    downsample_rate: int
    features: list[str]


class SleepSegTestDataset(Dataset):
    def __init__(
        self, cfg: TestSegDatasetConfig, chunk_features: dict[str, np.ndarray]
    ) -> None:
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(cfg.seq_len * cfg.upsample_rate), cfg.downsample_rate
        )

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]
        features = self.chunk_features[key]
        # (1, num_features, seq_len)
        feature = torch.FloatTensor(features.T).unsqueeze(0)
        feature = TF.resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        )
        feature = feature.squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, upsampled_num_frames)
        }


class DataloaderConfigV4(Protocol):
    data_dir: pathlib.Path
    processed_dir: pathlib.Path
    seed: int

    train_series: list[str]
    valid_series: list[str]

    sample_per_epoch: int | None

    seq_len: int
    """系列長の長さ"""
    features: list[str]

    batch_size: int

    upsample_rate: float
    """default: 1.0"""
    downsample_rate: int
    """default: 2"""

    bg_sampling_rate: float
    """negative labelのサンプリング率. default: 0.5"""
    offset: int
    """gaussian labelのoffset. default: 10"""
    sigma: int
    """gaussian labelのsigma. default: 10"""


def _init_test_dl(
    cfg,
    processed_dir: Path,
    seq_len: int,
    features: list[str],
    num_workers: int,
    seed: int,
    slide_size: int,
    do_min_max_normalize: bool,
) -> DataLoader:
    feature_dir = pathlib.Path(processed_dir)
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        seq_len=seq_len,
        feature_names=features,
        series_ids=series_ids,
        processed_dir=processed_dir,
        phase="test",
        slide_size=slide_size,
        do_min_max_normalize=do_min_max_normalize,
    )
    ds = SleepSegTestDataset(cfg, chunk_features)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=lambda _: my_utils.seed_everything(seed),
        # collate_fn=collate_fn,
        persistent_workers=True,  # data loaderが使い終わったProcessをkillしない
    )
    return dl


def _init_valid_dl(
    cfg,
    data_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    batch_size: int,
    seq_len: int,
    valid_series: list[str],
    features: list[str],
    num_workers: int,
    seed: int,
    slide_size: int,
    do_min_max_normalize: bool,
    use_corrected_events: bool,
    use_corrected_events_v2: bool,
) -> DataLoader:
    if use_corrected_events:
        logger.info("Use Corrected Events")
        event_df = pl.read_csv(data_dir / "train_events_corrected.csv").drop_nulls()
    elif use_corrected_events_v2:
        logger.info("Use Corrected Events V2")
        event_df = pl.read_csv(
            pathlib.Path("./input/for_train") / "train_events_v1130.csv"
        ).drop_nulls()
    else:
        logger.info("Use Original Events")
        event_df = pl.read_csv(data_dir / "train_events.csv").drop_nulls()
    valid_event_df = event_df.filter(pl.col("series_id").is_in(valid_series))
    valid_chunk_features = load_chunk_features(
        seq_len=seq_len,
        feature_names=features,
        series_ids=valid_series,
        processed_dir=processed_dir,
        phase="valid",
        slide_size=slide_size,
        do_min_max_normalize=do_min_max_normalize,
    )
    print("Valid", seq_len)
    ds = SleepSegValidDataset(cfg, valid_chunk_features, valid_event_df)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=lambda _: my_utils.seed_everything(seed),
        # collate_fn=collate_fn,
        persistent_workers=True,  # data loaderが使い終わったProcessをkillしない
    )
    return dl


def _init_train_dl(
    cfg,
    data_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    batch_size: int,
    seq_len: int,
    train_series: list[str],
    features: list[str],
    num_workers: int,
    seed: int,
    do_min_max_normalize: bool,
    use_corrected_events: bool,
    use_corrected_events_v2: bool,
    use_periodic_dict: bool,
    sample_per_epoch: int | None = None,
) -> DataLoader:
    if use_corrected_events:
        logger.info("Use Corrected Events")
        event_df = pl.read_csv(data_dir / "train_events_corrected.csv").drop_nulls()
    elif use_corrected_events_v2:
        logger.info("Use Corrected Events V2")
        event_df = pl.read_csv(
            pathlib.Path("./input/for_train") / "train_events_v1130.csv"
        ).drop_nulls()
    else:
        event_df = pl.read_csv(data_dir / "train_events.csv").drop_nulls()

    train_event_df = event_df.filter(pl.col("series_id").is_in(train_series))

    if use_periodic_dict:
        logger.info("Use periodic dict")
        train_series_df = (
            pl.read_csv(data_dir / "train_series.parquet")
            .filter(pl.col("series_id").is_in(train_series))
            .drop_nulls()
        ).to_pandas(use_pyarrow_extension_array=True)
        train_periodic_dict = my_utils.create_periodic_dict(train_series_df)
    else:
        train_periodic_dict = None

    train_features = load_features(
        feature_names=features,
        series_ids=train_series,
        processed_dir=processed_dir,
        phase="train",
        do_min_max_normalize=do_min_max_normalize,
    )
    ds = SleepSegTrainDataset(
        cfg,
        train_event_df,
        train_features,
        sample_per_epoch=sample_per_epoch,
        do_sleep_label_smoothing=getattr(cfg, "do_sleep_label_smoothing", False),
        train_periodic_dict=train_periodic_dict,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        # shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda _: my_utils.seed_everything(seed),
        # collate_fn=collate_fn,
        persistent_workers=True,  # data loaderが使い終わったProcessをkillしない
    )
    return dl


def init_dataloader(phase: str, cfg: DataloaderConfigV4) -> DataLoader:
    num_workers = 8 if mp.cpu_count() > 8 else 4

    if phase == "test":
        return _init_test_dl(
            cfg=cfg,
            processed_dir=cfg.processed_dir,
            seed=cfg.seed,
            num_workers=num_workers,
            seq_len=cfg.seq_len,
            features=cfg.features,
            slide_size=getattr(cfg, "slide_size", cfg.seq_len),
            do_min_max_normalize=getattr(cfg, "do_min_max_normalize", False),
        )

    if phase == "valid":
        return _init_valid_dl(
            cfg=cfg,
            seed=cfg.seed,
            num_workers=num_workers,
            batch_size=cfg.batch_size,
            valid_series=cfg.valid_series,
            seq_len=cfg.seq_len,
            features=cfg.features,
            data_dir=cfg.data_dir,
            processed_dir=cfg.processed_dir,
            slide_size=getattr(cfg, "slide_size", cfg.seq_len),
            do_min_max_normalize=getattr(cfg, "do_min_max_normalize", False),
            use_corrected_events=getattr(cfg, "use_corrected_events", False),
            use_corrected_events_v2=getattr(cfg, "use_corrected_events_v2", False),
        )

    # Train
    # featuresはscripts/prepare_data.pyで作成したもの
    return _init_train_dl(
        cfg=cfg,
        seed=cfg.seed,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        train_series=cfg.train_series,
        seq_len=cfg.seq_len,
        features=cfg.features,
        data_dir=cfg.data_dir,
        processed_dir=cfg.processed_dir,
        sample_per_epoch=cfg.sample_per_epoch,
        do_min_max_normalize=getattr(cfg, "do_min_max_normalize", False),
        use_corrected_events=getattr(cfg, "use_corrected_events", False),
        use_corrected_events_v2=getattr(cfg, "use_corrected_events_v2", False),
        use_periodic_dict=getattr(cfg, "use_train_periodic_dict", False),
    )


############################################
# test
############################################
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


def test_ds3():
    class Config:
        seed: int = 42
        batch_size = 32
        num_workers = 0
        # Used in build_dataloader
        window_size: int = 10
        root_dir: Path = Path(".")
        input_dir: Path = root_dir / "input"
        data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
        train_series_path: str | Path = (
            input_dir / "for_train" / "train_series_fold.parquet"
        )
        train_events_path: str | Path = data_dir / "train_events.csv"
        test_series_path: str | Path = data_dir / "test_series.parquet"

        normalize_type: str = "robust"
        seq_len: int = 1000
        offset_size: int = 250
        shift_size: int = 500

        out_size: int = 3

        series_dir = Path("./output/series")
        target_series_uni_ids_path: Path = series_dir / "target_series_uni_ids.pkl"

        sigma: int = 720
        downsample_factor: int = 12
        w_sigma: float = 0.15

    targets, series, ids = joblib.load(Config.target_series_uni_ids_path)
    ds = SleepDatasetV3(
        phase="train",
        targets=targets,
        data=series,
        ids=ids,
        sigma=Config.sigma,  # avg length of day is 24*60*12=17280
        downsample_factor=Config.downsample_factor,
        w_sigma=Config.w_sigma,
    )
    batch = ds[0]
    X, y = batch
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    print(X.shape)
    print(y.shape)
    print(y.max(), y.min(), y.mean(), y.std())

    import matplotlib.pyplot as plt

    print(targets[0])

    s = 300
    e = 1000
    plt.figure()
    plt.scatter(list(range(len(y[s:e]))), y[s:e, 0], label="onset", alpha=0.5)
    plt.scatter(list(range(len(y[s:e]))), y[s:e, 1], label="wakeup", alpha=0.5)
    plt.title("label dist.")
    plt.legend()
    plt.savefig(f"./output/analysis/test_ds3_batch0_label_{s}_{e}.png")
    plt.close("all")

    test_series = pl.read_parquet(Config.test_series_path)
    test = []
    for sid in test_series["series_id"].unique():
        test.append(test_series.filter(pl.col("series_id") == sid))

    ds_test = SleepDatasetV3(
        phase="test",
        targets=None,
        data=test,
        ids=test_series["series_id"].unique().to_list(),
        sigma=720,  # avg length of day is 24*60*12=17280
        downsample_factor=12,
        w_sigma=0.15,
    )
    batch = ds_test[0]
    X, y = batch
    assert isinstance(X, torch.Tensor)
    assert not isinstance(y, torch.Tensor)
    print(X.shape)
    print(y)


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

    dl = build_dataloader(config, 0, "train", debug=False)
    cnt = dict()
    for b in dl:
        # shape: (ln, 2)
        label: list[torch.Tensor] = b[1]
        for l in label:
            print(l)
            for li in l:
                if sum(li) == 0:
                    cnt[0] = cnt.get(0, 0) + 1
                else:
                    idx = int(li.argmax()) + 1
                    cnt[idx] = cnt.get(idx, 0) + 1
    print(cnt)

    print("build_dataloader test passed")


def test_build_dl_v2():
    class Config:
        seed: int = 42
        batch_size = 32
        num_workers = 0
        # Used in build_dataloader
        window_size: int = 10
        root_dir: Path = Path(".")
        input_dir: Path = root_dir / "input"
        data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
        train_series_path: str | Path = (
            input_dir / "for_train" / "train_series_fold.parquet"
        )
        train_events_path: str | Path = data_dir / "train_events.csv"
        test_series_path: str | Path = data_dir / "test_series.parquet"

        normalize_type: str = "robust"
        seq_len: int = 1000
        offset_size: int = 250
        shift_size: int = 500

        out_size: int = 3

        fe_dir: Path = Path("./output/fe_exp000")
        num_array_path: Path = fe_dir / "num_array.npy"
        target_array_path: Path = fe_dir / "target_array.npy"
        mask_array_path: Path = fe_dir / "mask_array.npy"
        pred_use_array_path: Path = fe_dir / "pred_use_array.npy"
        series_ids_array_path: Path = fe_dir / "series_ids_array.npy"
        time_array_path: Path = fe_dir / "time_array.npy"

    dl = build_dataloader_v2(Config, 0, "train", debug=True)
    # dl = build_dataloader_v2(Config, 0, "valid", debug=True)
    assert isinstance(dl, DataLoader)
    print(len(dl))
    batch = next(iter(dl))
    print(type(batch))
    print(len(batch))
    print(batch.keys())

    print(type(batch["input_data_num_array"]))
    print(len(batch["input_data_num_array"]))
    print(type(batch["input_data_mask_array"]))
    print(len(batch["input_data_mask_array"]))
    print(type(batch["attention_mask"]))
    print(len(batch["attention_mask"]))
    print(type(batch["y"]))
    print(len(batch["y"]))
    print(batch["y"].shape)

    print(type(batch["steps"]))
    print(batch["steps"].shape)
    print(batch["steps"][:5])

    print(type(batch["pred_use_array"]))
    print(batch["pred_use_array"].shape)
    print(batch["pred_use_array"][:5])

    print(type(batch["series_ids"]))
    print(len(batch["series_ids"]))
    print(batch["series_ids"][:5])

    print("build_dataloader test passed")


def test_build_dl_v3():
    class Config:
        seed: int = 42
        batch_size: int = 32
        num_workers: int = 0
        # Used in build_dataloader
        window_size: int = 10
        root_dir: Path = Path(".")
        input_dir: Path = root_dir / "input"
        data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
        train_series_path: str | Path = (
            input_dir / "for_train" / "train_series_fold.parquet"
        )
        train_events_path: str | Path = data_dir / "train_events.csv"
        test_series_path: str | Path = data_dir / "test_series.parquet"

        out_size: int = 3
        series_dir: Path = Path("./output/series")
        target_series_uni_ids_path: Path = series_dir / "target_series_uni_ids.pkl"

        sigma: int = 720
        downsample_factor: int = 12
        w_sigma: float = 0.15

        train_seq_len: int = 24 * 60 * 5

        random_sequence_mixing: bool = True
        sample_per_epoch: int = 20000

    dl = build_dataloader_v3(Config, 0, "train", debug=True)

    batch = next(iter(dl))
    print(type(batch))
    print(len(batch))
    x, y, sid, step = batch
    print(x.shape)
    print(y.shape)
    print(sid)
    print(step)


def test_build_dl_v4():
    class Config:
        seed: int = 42
        batch_size: int = 32
        num_workers: int = 0
        # Used in build_dataloader
        window_size: int = 10
        root_dir: Path = Path(".")
        input_dir: Path = root_dir / "input"
        data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
        train_series_path: str | Path = (
            input_dir / "for_train" / "train_series_fold.parquet"
        )
        train_events_path: str | Path = data_dir / "train_events.csv"
        test_series_path: str | Path = data_dir / "test_series.parquet"

        out_size: int = 3
        series_dir: Path = Path("./output/series")
        target_series_uni_ids_path: Path = series_dir / "target_series_uni_ids.pkl"

        data_dir: pathlib.Path = pathlib.Path(
            "./input/child-mind-institute-detect-sleep-states"
        )
        processed_dir: pathlib.Path = pathlib.Path("./input/processed")
        seed: int

        train_series: list[str] = [
            "3df0da2e5966",
            "05e1944c3818",
            "bfe41e96d12f",
            "062dbd4c95e6",
            "1319a1935f48",
            "67f5fc60e494",
            "d2d6b9af0553",
            "aa81faa78747",
            "4a31811f3558",
            "e2a849d283c0",
            "361366da569e",
            "2f7504d0f426",
            "e1f5abb82285",
            "e0686434d029",
            "6bf95a3cf91c",
            "a596ad0b82aa",
            "8becc76ea607",
            "12d01911d509",
            "a167532acca2",
        ]
        valid_series: list[str] = [
            "e0d7b0dcf9f3",
            "519ae2d858b0",
            "280e08693c6d",
            "25e2b3dd9c3b",
            "9ee455e4770d",
            "0402a003dae9",
            "78569a801a38",
            "b84960841a75",
            "1955d568d987",
            "599ca4ed791b",
            "971207c6a525",
            "def21f50dd3c",
            "8fb18e36697d",
            "51b23d177971",
            "c7b1283bb7eb",
            "2654a87be968",
            "af91d9a50547",
            "a4e48102f402",
        ]

        seq_len: int = 24 * 60 * 4
        """系列長の長さ"""
        features: list[str] = ["anglez", "enmo", "hour_sin", "hour_cos"]

        batch_size: int = 8

        upsample_rate: float = 1.0
        """default: 1.0"""
        downsample_rate: int = 2
        """default: 2"""

        bg_sampling_rate: float = 0.5
        """negative labelのサンプリング率. default: 0.5"""
        offset: int = 10
        """gaussian labelのoffset. default: 10"""
        sigma: int = 10
        """gaussian labelのsigma. default: 10"""
        sample_per_epoch: int | None = None

    with my_utils.trace("dataset load"):
        dl = init_dataloader("train", Config)
        # dl = init_dataloader("valid", Config)

    batch = next(iter(dl))
    print(type(batch))
    print(len(batch))
    print(batch.keys())

    print(batch["feature"].shape)
    print(batch["label"].shape)
    print(batch["series_id"])
    print(batch["weight"])

    for i in range(batch["feature"].shape[1]):
        print(
            i,
            batch["feature"][0, i, :].max(),
            batch["feature"][0, i, :].min(),
            batch["feature"][0, i, :].mean(),
            batch["feature"][0, i, :].std(),
            batch["feature"][0, i, :].median(),
        )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(batch["label"]), figsize=(20, 10))
    for i in range(len(batch["label"])):
        labels = batch["label"][i]
        sleep = labels[:, 0]
        onset = labels[:, 1]
        wakeup = labels[:, 2]
        # print(sleep[sleep > 0.0])
        # print(onset[onset > 0.0])
        # print(wakeup[wakeup > 0.0])
        # axes.plot(batch["feature"][b_idx, 0, :], label="anglez")  # type: ignore
        # axes.plot(batch["feature"][b_idx, 1, :], label="enmo")  # type: ignore
        # axes.plot(batch["feature"][b_idx, 2, :], label="hour_sin")  # type: ignore
        # axes.plot(batch["feature"][b_idx, 3, :], label="hour_cos")  # type: ignore
        axes[i].plot(sleep, label="sleep", alpha=0.5)  # type: ignore
        axes[i].plot(onset, label="onset", alpha=0.5)  # type: ignore
        axes[i].plot(wakeup, label="wakeup", alpha=0.5)  # type: ignore
        series_id = batch.get("series_id", batch["key"][i].split("_")[0])
        axes[i].set_title(f"sid: {series_id}")
        axes[i].legend()  # type: ignore
    plt.savefig("./output/eda/test_build_dl_v4.png")


def _test_load_chunk_features():
    class Config:
        seed: int = 42
        batch_size: int = 32
        num_workers: int = 0
        # Used in build_dataloader
        window_size: int = 10
        root_dir: Path = Path(".")
        input_dir: Path = root_dir / "input"
        data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
        train_series_path: str | Path = (
            input_dir / "for_train" / "train_series_fold.parquet"
        )
        train_events_path: str | Path = data_dir / "train_events.csv"
        test_series_path: str | Path = data_dir / "test_series.parquet"

        out_size: int = 3
        series_dir: Path = Path("./output/series")
        target_series_uni_ids_path: Path = series_dir / "target_series_uni_ids.pkl"

        data_dir: pathlib.Path = pathlib.Path(
            "./input/child-mind-institute-detect-sleep-states"
        )
        processed_dir: pathlib.Path = pathlib.Path("./input/processed")
        seed: int

        train_series: list[str] = [
            "3df0da2e5966",
            "05e1944c3818",
            "bfe41e96d12f",
            "062dbd4c95e6",
            "1319a1935f48",
            "67f5fc60e494",
            "d2d6b9af0553",
            "aa81faa78747",
            "4a31811f3558",
            "e2a849d283c0",
            "361366da569e",
            "2f7504d0f426",
            "e1f5abb82285",
            "e0686434d029",
            "6bf95a3cf91c",
            "a596ad0b82aa",
            "8becc76ea607",
            "12d01911d509",
            "a167532acca2",
        ]
        valid_series: list[str] = [
            "e0d7b0dcf9f3",
            "519ae2d858b0",
            "280e08693c6d",
            "25e2b3dd9c3b",
            "9ee455e4770d",
            "0402a003dae9",
            "78569a801a38",
            "b84960841a75",
            "1955d568d987",
            "599ca4ed791b",
            "971207c6a525",
            "def21f50dd3c",
            "8fb18e36697d",
            "51b23d177971",
            "c7b1283bb7eb",
            "2654a87be968",
            "af91d9a50547",
            "a4e48102f402",
        ]

        seq_len: int = 24 * 60 * 4
        """系列長の長さ"""
        features: list[str] = ["anglez", "enmo", "hour_sin", "hour_cos"]

        batch_size: int = 32

        upsample_rate: float = 1.0
        """default: 1.0"""
        downsample_rate: int = 2
        """default: 2"""

        bg_sampling_rate: float = 0.5
        """negative labelのサンプリング率. default: 0.5"""
        offset: int = 10
        """gaussian labelのoffset. default: 10"""
        sigma: int = 10
        """gaussian labelのsigma. default: 10"""
        sample_per_epoch: int | None = None

    cfg = Config()
    data_dir = Path(cfg.data_dir)
    processed_dir = Path(cfg.processed_dir)
    event_df = pl.read_csv(data_dir / "train_events.csv").drop_nulls()
    valid_event_df = event_df.filter(pl.col("series_id").is_in(cfg.valid_series))
    with my_utils.trace("load_chunk_features"):
        valid_chunk_features = load_chunk_features(
            seq_len=cfg.seq_len,
            feature_names=cfg.features,
            series_ids=cfg.valid_series,
            processed_dir=processed_dir,
            phase="valid",
        )


if __name__ == "__main__":
    # test_ds()
    # test_ds2()
    # test_ds3()
    # test_build_dl()
    # test_build_dl_v2()
    # test_build_dl_v3()
    # test_build_dl_v4()
    # _test_load_chunk_features()
    import numpy as np

    mask = make_periodic_mask(
        np.array([110, 111, 112, 113, 125, 126, 127]), 100, 120, 20
    )
    print(mask)
