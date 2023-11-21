import numpy as np
import pathlib

import matplotlib.pyplot as plt
from matplotlib import axes, figure
import polars as pl


DATA_DIR = pathlib.Path("./input/child-mind-institute-detect-sleep-states")
FOR_TRAIN_DIR = pathlib.Path("./input/for_train")
FEATURES_DIR = pathlib.Path("./input/processed")
USE_FEATURES = ["anglez", "enmo", "hour_sin", "hour_cos"]
pathlib.Path("./output/eda010").mkdir(parents=True, exist_ok=True)

# target_series_id = "038441c925bb"
target_series_id = "fe90110788d2"

df = pl.read_csv(FOR_TRAIN_DIR / "record_state.csv")
print(df)

event_df = pl.read_csv(DATA_DIR / "train_events.csv")
event_df = event_df.drop_nulls()
print(event_df)

features = {}
for feature_path in (FEATURES_DIR / target_series_id).glob("*.npy"):
    if feature_path.stem in USE_FEATURES:
        features[feature_path.stem] = np.load(feature_path)
feats_df = pl.DataFrame(features)


def plot(series_id):
    target_event_df = event_df.filter(pl.col("series_id") == series_id)
    target_df = df.filter(pl.col("series_id") == series_id).filter(pl.col("step") != 0)
    fig, ax = plt.subplots(len(feats_df.columns), 1, figsize=(20, 10))
    assert isinstance(ax, np.ndarray)
    assert isinstance(fig, figure.Figure)

    for feat, a in zip(feats_df.columns, ax):
        a.plot(feats_df[feat])
        a.set_title(feat)

    for row_id in range(len(target_df)):
        row = target_df[row_id].to_dict(as_series=False)
        color = "red" if row["awake"][0] == 1 else "green"
        label = "awake_corrected" if row["awake"][0] == 1 else "sleep_corrected"
        for a in ax:
            a.axvline(row["step"], color=color, alpha=0.5, label=label)

    for row_id in range(len(target_event_df)):
        row = target_event_df[row_id].to_dict(as_series=False)
        color = "yellow" if row["event"][0] == "onset" else "lightblue"
        label = "awake_original" if row["event"][0] == "onset" else "sleep_original"
        for a in ax:
            a.axvline(row["step"], color=color, alpha=0.5, label=label)

    for x in ax:
        handles, legends = x.get_legend_handles_labels()
        used_labels = set()
        unique_handles_legends = []
        for handle, legend in zip(handles, legends):
            if legend not in used_labels:
                unique_handles_legends.append((handle, legend))
                used_labels.add(legend)
        x.legend(*zip(*unique_handles_legends))

    fig.suptitle(series_id)
    fig.tight_layout()
    fig.savefig(f"./output/eda010/{series_id}.png")
    plt.close("all")


for series_id in df["series_id"].unique():
    print(series_id)
    plot(series_id)
