from pathlib import Path


class Config:
    name: str = "fe_exp001"

    root_dir: Path = Path(__file__).resolve().parents[2]
    input_dir: Path = root_dir / "input"
    data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
    output_dir: Path = root_dir / "output" / name
    output_dir.mkdir(exist_ok=True, parents=True)

    seed: int = 42

    # -- Additional params --

    # Used in build_dataloader
    window_size: int = 10
    # train_series_path: str | Path = data_dir / "train_series.parquet"
    train_series_path: str | Path = (
        input_dir / "for_train" / "train_series_fold.parquet"
    )
    train_events_path: str | Path = data_dir / "train_events.csv"
    test_series_path: str | Path = data_dir / "test_series.parquet"

    normalize_type: str = "robust"
    seq_len: int = 5000
    offset_size: int = 1250
    shift_size: int = 2500

    save_dir: Path = output_dir
    num_array_save_fname: str = "num_array.npy"
    target_array_save_fname: str = "target_array.npy"
    mask_array_save_fname: str = "mask_array.npy"
    pred_use_array_save_fname: str = "pred_use_array.npy"
    series_ids_array_save_fname: str = "series_ids_array.npy"
    time_array_save_fname: str = "time_array.npy"
