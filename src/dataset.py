from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import joblib
from src.fe import make_sequence_chunks
from src.utils import seed_everything

pl.Config.set_tbl_cols(n=300)


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
        self.series = series.reset_index()
        self.series_ids = series_ids
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
            worker_init_fn=lambda _: seed_everything(config.seed),
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
            worker_init_fn=lambda _: seed_everything(config.seed),
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
            worker_init_fn=lambda _: seed_everything(config.seed),
            # collate_fn=collate_fn,
            persistent_workers=True,  # data loaderが使い終わったProcessをkillしない
        )
        return dl_train


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


if __name__ == "__main__":
    # test_ds()
    # test_ds2()
    # test_ds3()
    # test_build_dl()
    # test_build_dl_v2()
    test_build_dl_v3()
