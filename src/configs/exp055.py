import pathlib
from pathlib import Path
from typing import Any

from src import utils


class Config:
    name: str = "exp055"
    desc: str = """
    wavegram. feature_extractor => encoder => decoder
    52+encoder=eca_nfnet_l1
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
    num_epochs: int = 10 * 4
    batch_size: int = int(8 * 2)
    num_workers: int = 8 * 1
    num_grad_accum: int = 2

    # Model
    model_type: str = "Spectrogram2DCNN"

    # criterion_type: str = "FocalLoss"
    criterion_type: str = "BCEWithLogitsLoss"
    optimizer_params: dict[str, Any] = dict(lr=5e-4, weight_decay=1e-2, eps=1e-4)
    scheduler_params: dict[str, Any] = dict(
        t_initial=num_epochs,
        lr_min=1e-6,
        warmup_prefix=True,
        warmup_t=0,
        warmup_lr_init=1e-7,
        cycle_limit=1,
    )
    early_stopping_params: dict[str, Any] = dict(
        patience=num_epochs, direction="minimize"
    )

    # -- Additional params --

    # Used in build_dataloader
    # train_series_path: str | Path = data_dir / "train_series.parquet"
    # train_series_path: str | Path = (
    #     input_dir / "for_train" / "train_series_fold.parquet"
    # )
    # train_events_path: str | Path = data_dir / "train_events.csv"
    # test_series_path: str | Path = data_dir / "test_series.parquet"

    train_events_path: str | Path = data_dir / "train_events.csv"

    processed_dir: pathlib.Path = input_dir / "processed"

    # seq_len: int = 24 * 60 * 4  # num_frames=seq_len//downsample_rate
    # offset: int = 720
    # sigma: int = 110
    offset: int = 10
    sigma: int = 10
    bg_sampling_rate: float = 0.5

    sample_per_epoch: int | None = None
    """SleepSegTrainDatasetの__len__で返される値。Noneの場合はlen(series_ids)."""

    # Train additional params
    mixup_prob: float = 0.0
    downsample_rate: int = 2
    upsample_rate: int = 1
    seq_len: int = 24 * 60 * 8
    # seq_len: int = 32 * 16 * 20
    # seq_len: int = 32 * 16 * 30

    fold: int = 0
    train_series: list[str] = utils.load_series(
        pathlib.Path("./input/for_train/folded_series_ids_fold5_seed42.json"),
        "train_series",
        fold,
    )
    valid_series: list[str] = utils.load_series(
        pathlib.Path("./input/for_train/folded_series_ids_fold5_seed42.json"),
        "valid_series",
        fold,
    )
    features: list[str] = [
        "anglez",
        "enmo",
        "hour_sin",
        "hour_cos",
    ]
    postprocess_params: dict[str, Any] = dict(
        score_thr=0.02,
        distance=90,
    )

    spectrogram2dcnn_params: dict[str, Any] = dict(
        downsample_rate=downsample_rate,
        # -- CNNSpectrogram
        in_channels=len(features),  # is same as feature_dim
        base_filters=64 * 1,
        kernel_size=[32, 16, downsample_rate],
        stride=downsample_rate,
        sigmoid=True,
        output_size=seq_len,
        # -- Unet1DDecoder
        n_classes=3,
        seq_len=seq_len,
        bilinear=False,
        se=False,
        res=False,
        scale_factor=2,
        dropout=0.0,
        # -- Spectrogram2DCNN
        # encoder_name="maxvit_rmlp_tiny_rw_256.sw_in1k",
        # encoder_name="tf_efficientnet_b0_ns",
        # encoder_name="resnet34",
        encoder_name="eca_nfnet_l1",
        encoder_weights="imagenet",
        use_sample_weights=False,
        use_spec_augment=False,
        spec_augment_params=dict(
            time_mask_param=100,
            freq_mask_param=10,
        ),
    )
