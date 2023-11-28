import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes, figure

from src import dataset, utils
from src.configs import exp070 as exp

pathlib.Path("./output/eda/eda012").mkdir(exist_ok=True, parents=True)

config = exp.Config()
# dl_train = dataset.init_dataloader("train", config)
dl_valid = dataset.init_dataloader("valid", config)


def _plot_feats_and_labels(
    feats: torch.Tensor, labels: torch.Tensor
) -> tuple[figure.Figure, np.ndarray]:
    """Plot feats and labels.

    Args:
        feats: (n_feats, n_frames)
        labels: (n_frames, 3)

    Returns:
        tuple[figure.Figure, axes.Axes]: fig, ax
    """

    fig, ax = plt.subplots(len(feats) + 1, 1, figsize=(20, 5))
    assert isinstance(ax, np.ndarray)
    assert isinstance(fig, figure.Figure)
    feats = feats.detach().cpu().numpy()

    for i, feat in enumerate(feats):
        ax[i].plot(feat)
        ax[i].set_title(f"feat {i}")
        ax[i].grid()

    for i in range(3):
        color = {0: "red", 1: "green", 2: "blue"}[i]
        label = {0: "sleep", 1: "onset", 2: "wakeup"}[i]
        ax[-1].plot(labels[:, i], label=label, color=color)
        ax[-1].set_title("labels")
        ax[-1].grid()

    fig.tight_layout()
    fig.legend()
    return fig, ax


for batch in dl_valid:
    for i in range(len(batch["feature"])):
        feats = batch["feature"][i]
        labels = batch["label"][i]
        fig, ax = _plot_feats_and_labels(feats, labels)
        fig.savefig(f"./output/eda/eda012/eda012_{i}.png")
    break
