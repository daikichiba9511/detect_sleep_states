from pathlib import Path

import polars as pl


class CFG:
    name = "eda002"

    data_dir = Path("./input/child-mind-institute-detect-sleep-states")
    train_event_path = data_dir / "train_events.csv"
    train_series_path = data_dir / "train_series.parquet"
    sample_submission_path = data_dir / "sample_submission.csv"
    test_series_path = data_dir / "test_series.parquet"

    output_dir = Path(f"./output/eda/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)


df_test_series = pl.read_parquet(CFG.test_series_path)
print(df_test_series)
