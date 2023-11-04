from pathlib import Path
from typing import Any


class Config:
    name: str = "exp019"
    desc: str = """
    exp017 + SleepTransformer
    """

    root_dir: Path = Path(__file__).resolve().parents[2]
    input_dir: Path = root_dir / "input"
    data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
    output_dir: Path = root_dir / "output" / name
    output_dir.mkdir(exist_ok=True, parents=True)

    seed: int = 42

    model_save_name = f"{name}_model_fold"
    model_save_path: Path = output_dir / (model_save_name + "0.pth")
    metrics_save_path: Path = output_dir / f"{name}_metrics.csv"
    metrics_plot_path: Path = output_dir / f"{name}_losses.png"

    # Train
    use_amp: bool = True
    num_epochs: int = 10 * 5
    batch_size: int = 8 * 3
    num_workers: int = 8 * 2

    criterion_type: str = "MSELoss"
    # criterion_type: str = "BCEWithLogitsLossWeightedPos"
    optimizer_params: dict[str, Any] = dict(lr=1e-2, weight_decay=1e-2, eps=1e-4)
    scheduler_params: dict[str, Any] = dict(
        t_initial=num_epochs,
        lr_min=1e-6,
        warmup_prefix=True,
        warmup_t=1,
        warmup_lr_init=1e-7,
    )
    early_stopping_params: dict[str, Any] = dict(patience=5, direction="minimize")

    # -- Additional params --

    # Used in build_dataloader
    # train_series_path: str | Path = data_dir / "train_series.parquet"
    train_series_path: str | Path = (
        input_dir / "for_train" / "train_series_fold.parquet"
    )
    train_events_path: str | Path = data_dir / "train_events.csv"
    test_series_path: str | Path = data_dir / "test_series.parquet"

    # Used in build_model
    model_type: str = "SleepTransformer"
    # input_size: int = 18
    input_size: int = 10
    hidden_size: int = 64 * 2
    # model_size: int = 128
    model_size: int = 10
    linear_out: int = 128
    out_size: int = 2
    n_layers: int = 5

    num_grad_accum: int = 8

    bidir: bool = True

    # Used in train_one_epoch_v2, valid_one_epoch_v2
    train_chunk_size: int = 24 * 60  # 1hour
    infer_chunk_size: int = 24 * 60 * 100  # 100hour
    """推論時のchunk_size"""
    train_seq_len: int = 24 * 60 * 5  # 24 * 60 = 1440 min / day
    """train時にdatasetで、どの長さのseriesを切り出すか"""
    # infer_seq_len: int = 24 * 60 * 6  # 6hour
    series_save_dir: Path = root_dir / "output" / "series"
    series_save_dir.mkdir(exist_ok=True, parents=True)
    target_series_uni_ids_path: Path = series_save_dir / "target_series_uni_ids.pkl"

    sigma: int = 720
    w_sigma: float = 0.15
    downsample_factor: int = 12

    random_sequence_mixing: bool = True
    sample_per_epoch: int = 5000
    # sample_per_epoch: int = 25

    transformer_params: dict[str, Any] = dict(
        model_dim=10,
        embed_dim=320,
        seq_model_dim=320,
        num_heads=8,
        num_encoder_layers=5,
        num_lstm_layers=2,
        dropout=0.0,
        seq_len=train_seq_len,
        fc_hidden_dim=64 * 2,
    )
