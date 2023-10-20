from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import json

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


if __name__ == "__main__":
    # test_ds()
    # test_ds2()
    # test_build_dl()
    test_build_dl_v2()
