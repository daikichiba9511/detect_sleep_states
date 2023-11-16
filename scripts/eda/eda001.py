"""
check data size/infos
"""

from pathlib import Path

import polars as pl


class CFG:
    name = "eda001"

    data_dir = Path("./input/child-mind-institute-detect-sleep-states")
    train_event_path = data_dir / "train_events.csv"
    train_series_path = data_dir / "train_series.parquet"
    sample_submission_path = data_dir / "sample_submission.csv"
    test_series_path = data_dir / "test_series.parquet"

    output_dir = Path(f"./output/eda/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)


df_train_events = pl.read_csv(CFG.train_event_path)
print(df_train_events)
assert len(df_train_events) == 14508
print(f"{df_train_events.group_by('series_id').count().shape[0]} series")  # 277
print(f"{df_train_events.group_by('series_id').count()}")
df_trn_cnt_events = df_train_events.group_by("series_id").count()
df_trn_cnt_events.to_pandas().hist()
print(
    f"{df_trn_cnt_events.min() = } \n {df_trn_cnt_events.max() = } \n {df_trn_cnt_events.mean() = } \n {df_trn_cnt_events.std() = } \n {df_trn_cnt_events.median() = }"
)


df_train_series = pl.read_parquet(CFG.train_series_path)
print(df_train_series)
assert len(df_train_series) == 127_946_340
print(f"{df_train_series.group_by('series_id').count().shape[0]} series")  # 277
print(f"{df_train_series.group_by('series_id').count()}")
df_trn_cnt_series = df_train_series.group_by("series_id").count()
print(
    f"{df_trn_cnt_series.min() = } \n {df_trn_cnt_series.max() = } \n {df_trn_cnt_series.mean() = } \n {df_trn_cnt_series.std() = } \n {df_trn_cnt_series.median() = }"
)

# series_cnt_max=>143880step, series_id=fe90110788d2
# series_cnt_min=>37080step, series_id=038441c925bb

print(df_train_events.filter(pl.col("series_id") == "fe90110788d2"))
print(df_train_series.filter(pl.col("series_id") == "fe90110788d2"))
print(df_train_events.filter(pl.col("series_id") == "038441c925bb"))
print(df_train_series.filter(pl.col("series_id") == "038441c925bb"))

# eventsにstepが欠損してるのが存在するのでseriesとmappingを取れないサンプルが存在する
# どれぐらいあるか？

print(
    df_train_events.filter(pl.col("step").is_null())["series_id"].n_unique()
)  # -> 240

print(df_train_events.group_by("event").count())

print(
    f"""
    {df_train_series["step"].min() = }
    {df_train_series["step"].max() = }
    {df_train_series["step"].mean() = }
    {df_train_series["step"].std() = }
    {df_train_series["step"].median() = }
    """
)

print(
    df_train_series.filter(pl.col("series_id") == "038441c925bb")
    .with_columns(
        [
            pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
        ]
    )
    .with_columns(
        [
            pl.col("timestamp").dt.day().alias("day"),
            pl.col("timestamp").dt.month().alias("month"),
        ]
    )
    .filter((pl.col("month") == 8) & (pl.col("day") == 16))
)

# nullを含んでるseries_idと含んでないのに分ける
df_train_events_null = df_train_events.filter(pl.col("step").is_null())[
    "series_id"
].unique()
df_train_events_not_null = df_train_events.filter(pl.col("step").is_not_null())[
    "series_id"
].unique()
print(df_train_events_null)
print(df_train_events_not_null)
import json

with open("./output/series_ids.json", "w") as fp:
    json.dump(
        {
            "non_null": df_train_events_not_null.to_list(),
            "null": df_train_events_null.to_list(),
        },
        fp=fp,
        indent=4,
        ensure_ascii=False,
    )
