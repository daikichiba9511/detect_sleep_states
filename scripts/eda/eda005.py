from pathlib import Path

import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)


class Config:
    name: str = "eda005"
    root_dir: Path = Path(".")
    input_dir: Path = root_dir / "input"
    data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
    train_series_path: str | Path = (
        input_dir / "for_train" / "train_series_fold.parquet"
    )
    output_dir: Path = root_dir / "output" / name
    output_dir.mkdir(exist_ok=True, parents=True)
    train_event_path: Path = input_dir / ""
    train_events_path: str | Path = data_dir / "train_events.csv"
    num_array_save_fname: str = "num_array.npy"
    target_array_save_fname: str = "target_array.npy"
    mask_array_save_fname: str = "mask_array.npy"
    pred_use_array_save_fname: str = "pred_use_array.npy"
    series_ids_array_save_fname: str = "series_ids_array.npy"

    series_save_dir: Path = root_dir / "output" / "series"
    series_save_dir.mkdir(exist_ok=True, parents=True)


df_series = pl.read_parquet(Config.train_series_path)
df_series = df_series.sort(by=["series_id", "step"])
df_series = df_series.with_columns(
    pl.arange(0, df_series.height).alias("idx"),
)
df_events = pl.read_csv(Config.train_events_path)
print("Before len(events): ", len(df_events))
df_events = df_events.drop_nulls(subset=["timestamp"])
print("After len(events): ", len(df_events))
series_ids = df_series["series_id"].unique().to_list()

targets: list[list[tuple[int, int]]] = []
series: list[pl.DataFrame] = []
target_ids = []
for target_id in series_ids:
    target_series = df_series.filter(pl.col("series_id") == target_id).clone()
    target_series = target_series.with_columns(
        pl.col("timestamp")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z", time_unit="ns")
        .alias("dt")
    ).with_columns(
        pl.col("dt").dt.hour().alias("hour"),
        pl.arange(0, len(target_series)).alias("local_idx"),
    )

    # print(target_series)

    target_events = df_events.filter(pl.col("series_id") == target_id)

    # targetのeventを含むseriesの範囲を取得する
    target_ranges: list[tuple[int, int]] = []
    check = 0
    for i in range(len(target_events) - 1):
        # eventのdfはonset/wakeupの順番に交互に並んでいる
        # 同一の日のonset/wakeupのpairを取得する
        occur_onset = target_events[i]["event"] == "onset"
        occur_wakeup = target_events[i + 1]["event"] == "wakeup"
        is_same_night = target_events[i]["night"] == target_events[i + 1]["night"]
        # すべて揃っているときにrangeを取得する
        if occur_onset[0] and occur_wakeup[0] and is_same_night[0]:
            start = target_events[i]["timestamp"]
            end = target_events[i + 1]["timestamp"]
            start_id = target_series.filter(pl.col("timestamp") == start)["local_idx"]
            end_id = target_series.filter(pl.col("timestamp") == end)["local_idx"]
            target_ranges.append((start_id[0], end_id[0]))
            check += 1
    targets.append(target_ranges)
    series.append(
        target_series.select([pl.col("anglez", "enmo", "step", "idx", "local_idx")])
    )
    target_ids.append(target_id)

print(targets)
print(series)

target_i = targets[0][0]
print(target_i)
# print(series[0].filter(target_i[0] < pl.col("idx")).filter(pl.col("idx") < target_i[1]))
print(
    series[0].filter(
        (target_i[0] < pl.col("local_idx")) & (pl.col("local_idx") < target_i[1])
    )
)

import joblib

joblib.dump(
    (targets, series, series_ids), Config.series_save_dir / "target_series_uni_ids.pkl"
)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(20, 5))
assert isinstance(fig, plt.Figure)
assert isinstance(ax, np.ndarray)

for i in range(5):
    data_i = series[i]
    ax[0].plot(
        data_i["local_idx"].to_numpy(),
        data_i["anglez"].to_numpy(),
        label="anglez",
        color="blue",
    )
    ax[0].set_ylabel("anglez")
    ax[0].set_xlabel("step")
    ax[0].set_title(f"series_id: {target_ids[0]}")

    ax[1].plot(
        data_i["local_idx"].to_numpy(),
        data_i["enmo"].to_numpy(),
        label="enmo",
        color="orange",
    )
    ax[1].set_xlabel("step")
    ax[0].set_ylabel("enmo")
    ax[0].set_title(f"series_id: {target_ids[0]}")

    for start, end in targets[i]:
        ax[0].axvspan(start, end, color="green", alpha=0.3)
        ax[1].axvspan(start, end, color="green", alpha=0.3)

    ax[0].legend()
    ax[1].legend()
    fig.savefig(Config.output_dir / f"target_series_uni_sample_{i}.png")
