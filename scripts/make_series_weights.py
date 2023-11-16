import json
import pathlib

import polars as pl


class CFG:
    data_dir = pathlib.Path("./input/child-mind-institute-detect-sleep-states")
    train_event_path = data_dir / "train_events.csv"
    train_series_path = data_dir / "train_series.parquet"
    sample_submission_path = data_dir / "sample_submission.csv"
    test_series_path = data_dir / "test_series.parquet"

    output_dir = pathlib.Path("./output/series_weights")
    output_dir.mkdir(parents=True, exist_ok=True)


"""
eventのseries_idごとのnullの数を調べる。
"""

event_df = pl.read_csv(CFG.train_event_path)
null_rate_df = event_df.group_by("series_id").agg(
    (pl.col("step").is_null().sum() / pl.col("series_id").count()).alias(
        "step_null_rate"
    ),
    (pl.col("timestamp").is_null().sum() / pl.col("series_id").count()).alias(
        "timestamp_null_rate"
    ),
)
null_rate_dict = null_rate_df.to_dict(as_series=False)
null_rate_dict = [
    {"series_id": s, "step_null_rate": snr, "timestamp_null_rate": tnr}
    for s, snr, tnr in zip(
        null_rate_dict["series_id"],
        null_rate_dict["step_null_rate"],
        null_rate_dict["timestamp_null_rate"],
    )
]
series_weights = [
    {
        "series_id": d["series_id"],
        "weight": 1.0 - ((d["step_null_rate"] + d["timestamp_null_rate"]) / 2),
        "step_null_rate": d["step_null_rate"],
        "timestamp_null_rate": d["timestamp_null_rate"],
    }
    for d in null_rate_dict
]
print(series_weights)

with (CFG.output_dir / "series_weights.json").open("w") as f:
    json.dump(series_weights, f, indent=4)
