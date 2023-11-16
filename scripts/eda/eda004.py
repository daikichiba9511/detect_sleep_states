import json
from pathlib import Path

import polars as pl


class Config:
    root_dir: Path = Path(".")
    input_dir: Path = root_dir / "input"
    data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
    train_series_path: str | Path = (
        input_dir / "for_train" / "train_series_fold.parquet"
    )
    train_event_path: Path = input_dir / ""
    train_events_path: str | Path = data_dir / "train_events.csv"
    save_dir: Path = Path("./output/fe_exp000")
    num_array_save_fname: str = "num_array.npy"
    target_array_save_fname: str = "target_array.npy"
    mask_array_save_fname: str = "mask_array.npy"
    pred_use_array_save_fname: str = "pred_use_array.npy"
    series_ids_array_save_fname: str = "series_ids_array.npy"

    series_save_dir: Path = root_dir / "output" / "series"
    series_save_dir.mkdir(exist_ok=True, parents=True)


def main():
    df_train_series = pl.read_parquet(Config.train_series_path)
    print(df_train_series)
    df_train_events = pl.read_csv(Config.train_events_path)
    print(df_train_events)

    uni_series_ids_has_label = df_train_events["series_id"].unique()
    print(uni_series_ids_has_label)
    series_ids_has_null_step = df_train_events.filter(pl.col("step").is_null())[
        "series_id"
    ].unique()
    print(series_ids_has_null_step)

    uni_series_does_not_have_null_step_label = df_train_events.filter(
        pl.col("series_id").is_in(series_ids_has_null_step).not_()
    )["series_id"].unique()
    print(uni_series_does_not_have_null_step_label)

    df_train_series = df_train_series.cast({"step": pl.Int64})
    df_train_events = df_train_events.cast({"step": pl.Int64})
    df_train_series = df_train_series.join(df_train_events, on=["series_id", "step"])

    series_labels = dict(
        have_null=series_ids_has_null_step.to_list(),
        not_have_null=uni_series_does_not_have_null_step_label.to_list(),
    )
    with (Config.series_save_dir / "series_id_null_label.json").open("w") as fp:
        json.dump(series_labels, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
