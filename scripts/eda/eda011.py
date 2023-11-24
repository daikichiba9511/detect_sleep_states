import numpy as np
import pathlib

import matplotlib.pyplot as plt
from matplotlib import axes, figure
import polars as pl

from src import utils

pl.Config.set_tbl_rows(1000)


DATA_DIR = pathlib.Path("./input/child-mind-institute-detect-sleep-states")
FOR_TRAIN_DIR = pathlib.Path("./input/for_train")
FEATURES_DIR = pathlib.Path("./input/processed")
USE_FEATURES = ["anglez", "enmo", "hour_sin", "hour_cos"]
pathlib.Path("./output/eda010").mkdir(parents=True, exist_ok=True)

CORRECTED_EVENT_DF = pl.read_csv(FOR_TRAIN_DIR / "record_state.csv")
# print(CORRECTED_EVENT_DF)

EVENT_DF = pl.read_csv(DATA_DIR / "train_events.csv")
# print(EVENT_DF)

# print(
#     EVENT_DF.pivot(index=["series_id", "night"], columns="event", values="step")
#     .drop_nulls()
#     .to_pandas(use_pyarrow_extension_array=True)
# )
# print(EVENT_DF["night"].value_counts())
# print(EVENT_DF["night"].is_null().sum())

df = utils.transformed_record_state(CORRECTED_EVENT_DF)
print(df)
print(
    df.pivot(
        index=["series_id", "night"],
        columns="event",
        values="step",
        # aggregate_function="max",
        aggregate_function=None,
    )
    # .drop_nulls()
    .to_pandas(use_pyarrow_extension_array=True)
)
raise Exception

# df = EVENT_DF

df = df.with_columns(
    pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
).with_columns(
    pl.col("timestamp").dt.month().alias("month"),
    pl.col("timestamp").dt.day().alias("day"),
    pl.col("timestamp").dt.hour().alias("hour"),
)
print(df.filter(pl.col("series_id") == "fcca183903b7"))

dfs = []
for series_id, series_df in df.groupby("series_id"):
    print("****** SeriesDF ******")
    print("Before unique:", len(series_df))
    serires_df = series_df.unique(subset=["event", "step"])
    print("After unique:", len(series_df))
    series_df = series_df.sort("step").with_columns(
        pl.Series("night", np.arange(len(series_df)) // 2 + 1)
    )
    print(series_df)
    dfs.append(series_df)

print("****** Concat ******")
df = pl.concat(dfs).sort(by=["series_id", "night"])
print(df)

df_pivot = (
    df.pivot(
        index=["series_id", "night"],
        columns="event",
        values="step",
        aggregate_function="min",
    )
    .drop_nulls()
    .to_pandas(use_pyarrow_extension_array=True)
)

print(df_pivot)
print(df_pivot.at[0, "onset"])
