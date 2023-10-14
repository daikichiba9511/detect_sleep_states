from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Config:
    name: str = "exp000"

    root_dir: Path = Path(__file__).resolve().parents[2]
    input_dir: Path = root_dir / "input"
    data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
    output_dir: Path = root_dir / "output" / name
    output_dir.mkdir(exist_ok=True, parents=True)

    seed: int = 42

    model_save_path: Path = output_dir / f"{name}_model.pth"
    metrics_save_path: Path = output_dir / f"{name}_metrics.csv"
    metrics_plot_path: Path = output_dir / f"{name}_losses.png"

    # Train
    use_amp: bool = True
    num_epochs: int = 10
    batch_size: int = 1
    num_workers: int = 16

    criterion_type: str = "BCEWithLogitsLoss"
    optimizer_params: dict[str, Any] = dict(lr=1e-4, weight_decay=1e-2, eps=1e-4)
    scheduler_params: dict[str, Any] = dict(
        t_initial=num_epochs,
        lr_min=1e-6,
        warmup_prefix=True,
        warmup_t=1,
        warmup_lr_init=1e-7,
    )
    early_stopping_params: dict[str, Any] = dict(patience=10, direction="minimize")

    # -- Additional params --

    # Used in build_dataloader
    window_size: int = 10
    # train_series_path: str | Path = data_dir / "train_series.parquet"
    train_series_path: str | Path = (
        input_dir / "for_train" / "train_series_fold.parquet"
    )
    train_events_path: str | Path = data_dir / "train_events.csv"
    test_series_path: str | Path = data_dir / "test_series.parquet"

    # Used in build_model
    model_type: str = "MultiResidualBiGRU"
    input_size: int = window_size
    hidden_size: int = 64
    out_size: int = 2
    n_layers: int = 5
    bidir: bool = True
