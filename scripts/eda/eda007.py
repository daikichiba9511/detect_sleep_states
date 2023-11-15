from typing import Any
from src import dataset, models, utils, feature_extractors, decoders
import pathlib
import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
from matplotlib import figure, axes

# device = torch.device("cuda")
device = torch.device("cpu")
seq_len = 24 * 60 * 4
downsample_rate = 2

cfg = importlib.import_module("src.configs.exp035").Config
params: dict[str, Any] = cfg.spectrogram2dcnn_params

# params: dict[str, Any] = dict(
#     downsample_rate=downsample_rate,
#     # -- CNNSpectgram
#     in_channels=4,
#     base_filters=64,
#     kernel_size=[32, 16, downsample_rate],
#     stride=downsample_rate,
#     sigmoid=True,
#     output_size=seq_len,
#     # -- Unet1DDecoder
#     n_classes=3,
#     duration=seq_len,
#     bilinear=False,
#     se=False,
#     res=False,
#     scale_factor=2,
#     dropout=0.2,
#     # -- Spectrogram2DCNN
#     # encoder_name="resnet18",
#     # encoder_name="timm-maxvit_tiny_tf_224.in1k",  # これ動かすにはカスタム必要
#     encoder_name="mit_b3",
#     # encoder_weights="imagenet",
#     encoder_weights=None,
# )

feature_extractor = feature_extractors.CNNSpectgram(
    in_channels=params["in_channels"],
    base_filters=params["base_filters"],
    kernel_size=params["kernel_size"],
    stride=params["stride"],
    sigmoid=params["sigmoid"],
    output_size=params["output_size"] // params["downsample_rate"],
)
decoder = decoders.Unet1DDecoder(
    n_channels=feature_extractor.height,
    n_classes=params["n_classes"],
    duration=params["seq_len"] // params["downsample_rate"],
    bilinear=params["bilinear"],
    se=params["se"],
    res=params["res"],
    scale_factor=params["scale_factor"],
    dropout=params["dropout"],
)
model = models.Spectrogram2DCNN(
    feature_extractor,
    decoder,
    encoder_name=params["encoder_name"],
    in_channels=feature_extractor.out_chans,
    encoder_weights=params["encoder_weights"],
    use_sample_weights=True,
    use_custom_encoder=False,
)
model = model.to(device).train()
print(model.load_state_dict(torch.load("./output/exp035/exp035_model_fold0.pth")))

num_features = 4
bs = 8 * 2


class Config:
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0
    # Used in build_dataloader
    window_size: int = 10
    root_dir: pathlib.Path = pathlib.Path(".")
    input_dir: pathlib.Path = root_dir / "input"
    data_dir: pathlib.Path = input_dir / "child-mind-institute-detect-sleep-states"
    train_series_path: str | pathlib.Path = (
        input_dir / "for_train" / "train_series_fold.parquet"
    )
    train_events_path: str | pathlib.Path = data_dir / "train_events.csv"
    test_series_path: str | pathlib.Path = data_dir / "test_series.parquet"

    out_size: int = 3
    series_dir: pathlib.Path = pathlib.Path("./output/series")
    target_series_uni_ids_path: pathlib.Path = series_dir / "target_series_uni_ids.pkl"

    data_dir: pathlib.Path = pathlib.Path(
        "./input/child-mind-institute-detect-sleep-states"
    )
    processed_dir: pathlib.Path = pathlib.Path("./input/processed")

    train_series: list[str] = [
        "3df0da2e5966",
        "05e1944c3818",
        "bfe41e96d12f",
        "062dbd4c95e6",
        "1319a1935f48",
        "67f5fc60e494",
        "d2d6b9af0553",
        "aa81faa78747",
        "4a31811f3558",
        "e2a849d283c0",
        "361366da569e",
        "2f7504d0f426",
        "e1f5abb82285",
        "e0686434d029",
        "6bf95a3cf91c",
        "a596ad0b82aa",
        "8becc76ea607",
        "12d01911d509",
        "a167532acca2",
    ]
    valid_series: list[str] = [
        "e0d7b0dcf9f3",
        "519ae2d858b0",
        "280e08693c6d",
        "25e2b3dd9c3b",
        "9ee455e4770d",
        "0402a003dae9",
        "78569a801a38",
        "b84960841a75",
        "1955d568d987",
        "599ca4ed791b",
        "971207c6a525",
        "def21f50dd3c",
        "8fb18e36697d",
        "51b23d177971",
        "c7b1283bb7eb",
        "2654a87be968",
        "af91d9a50547",
        "a4e48102f402",
    ]

    seq_len: int = 24 * 60 * 4
    """系列長の長さ"""
    features: list[str] = ["anglez", "enmo", "hour_sin", "hour_cos"]

    batch_size: int = 8

    upsample_rate: float = 1.0
    """default: 1.0"""
    downsample_rate: int = 2
    """default: 2"""

    bg_sampling_rate: float = 0.5
    """negative labelのサンプリング率. default: 0.5"""
    offset: int = 10
    """gaussian labelのoffset. default: 10"""
    sigma: int = 10
    """gaussian labelのsigma. default: 10"""
    sample_per_epoch: int | None = None


## Main
dl = dataset.init_dataloader("train", Config)
batch = next(iter(dl))
x = batch["feature"].to(device)
y = batch["label"].to(device)

# x = torch.randn(bs, num_features, seq_len).to(device)
# print(x.shape)
with utils.trace("model forward"):
    # p = model(x, y)
    # print(type(p))
    # print(p["logits"].shape)
    # print(p["loss"].shape)
    # print(p["loss"])
    wavegrams = feature_extractor(x)
print(x.shape, y.shape, wavegrams.shape)


def plot_wavegrame_with_raw_signal(
    wavegram: torch.Tensor, label: torch.Tensor
) -> tuple[figure.Figure, np.ndarray]:
    """Plot wavegram and raw_signal.

    Args:
        wavegram: (3, out_chans, seq_len)
        label: (seq_len//2, n_classes)

    """
    wg = wavegram.permute(1, 2, 0).detach().cpu().numpy()
    # label = label.detach().cpu().numpy()

    chunk_size = 500
    num_chunk = wg.shape[1] // chunk_size
    print(num_chunk, wg.shape)

    fig, ax = plt.subplots(num_chunk, 1, figsize=(20, 10))
    assert isinstance(ax, np.ndarray)
    # assert isinstance(ax, axes.Axes)
    assert isinstance(fig, figure.Figure)

    for i in range(num_chunk - 1):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, wg.shape[1])
        wg_chunk = wg[:, start:end, :]
        # print(wg_chunk.shape)
        ax[i].imshow(wg_chunk, label="wavegram")

    # axes[0].imshow(wg, label="wavegram")
    # axes[1].plot(label[..., 0], label="sleep")
    # axes[1].plot(label[..., 1], label="onset")
    # axes[1].plot(label[..., 2], label="wakeup")
    # fig.legend()
    fig.tight_layout()
    return fig, ax


save_dir = pathlib.Path("./output/eda/eda007")
save_dir.mkdir(exist_ok=True, parents=True)
for i, (wavegram, label) in enumerate(zip(wavegrams, y)):
    fig, _ = plot_wavegrame_with_raw_signal(wavegram, label)
    fig.savefig(save_dir / f"wavegram_{i}.png")
    plt.close("all")
