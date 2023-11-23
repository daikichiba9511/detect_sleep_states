import numpy as np
import pathlib

import matplotlib.pyplot as plt
from matplotlib import axes, figure
import polars as pl

from src import utils


DATA_DIR = pathlib.Path("./input/child-mind-institute-detect-sleep-states")
FOR_TRAIN_DIR = pathlib.Path("./input/for_train")
FEATURES_DIR = pathlib.Path("./input/processed")
USE_FEATURES = ["anglez", "enmo", "hour_sin", "hour_cos"]
pathlib.Path("./output/eda010").mkdir(parents=True, exist_ok=True)

CORRECTED_EVENT_DF = pl.read_csv(FOR_TRAIN_DIR / "record_state.csv")
print(CORRECTED_EVENT_DF)

EVENT_DF = pl.read_csv(DATA_DIR / "train_events.csv")
print(EVENT_DF)

print(
    EVENT_DF.pivot(index=["series_id", "night"], columns="event", values="step")
    .drop_nulls()
    .to_pandas(use_pyarrow_extension_array=True)
)
print(EVENT_DF["night"].value_counts())
print(EVENT_DF["night"].is_null().sum())

df = utils.transformed_record_state(CORRECTED_EVENT_DF)
print(df)

dfs = []
for series_id, series_df in df.groupby("series_id"):
    series_df = series_df.with_columns(
        pl.Series("night", np.arange(len(series_df)) // 2 + 1)
    )
    print(series_df)
    dfs.append(series_df)
df = pl.concat(dfs).sort(by=["series_id", "night"])
print(df)

print(
    df.pivot(
        index=["series_id", "night"],
        columns="event",
        values="step",
        aggregate_function="min",
    )
    .drop_nulls()
    .to_pandas(use_pyarrow_extension_array=True)
)
