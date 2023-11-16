import argparse
import copy
import importlib
import pathlib
import pprint
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import axes, figure

from src import metrics
from src.run import Runner
from src.utils import LoggingUtils, get_class_vars

logger = LoggingUtils.get_stream_logger(20)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="exp000")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--all", default=False, action="store_true")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

config = importlib.import_module(f"src.configs.{args.config}").Config

####### UPDATE PARAMS #######
config.fold = args.fold
config.model_save_path = config.output_dir / f"{config.name}_model_fold{args.fold}.pth"


logger.info(f"fold: {args.fold}, debug: {args.debug}")
logger.info(f"\n{pprint.pformat(get_class_vars(config))}")

########## VALID ##########
# df_valid_series = pl.read_parquet(config.train_series_path).filter(
#     pl.col("fold") == args.fold
# )
df_valid_series = config.valid_series
df_valid_events = pl.read_csv(config.train_events_path, dtypes={"step": pl.Int64})
df_valid_solution = df_valid_events.filter(
    pl.col("series_id").is_in(df_valid_series)
).to_pandas(use_pyarrow_extension_array=True)
df_valid_solution = df_valid_solution[~df_valid_solution["step"].isnull()]
print(df_valid_solution)

configs = []
if args.all:
    config = importlib.import_module(f"src.configs.{args.config}").Config
    config_ = config()
    config_.fold = args.fold
    config_.model_save_path = (
        config_.output_dir / f"{config_.name}_model_fold{args.fold}.pth"
    )
    # logger.info(
    #     "model_path: {model_path}".format(model_path=config_.model_save_path)
    # )
    configs.append(copy.deepcopy(config_))
else:
    configs.append(config)

print(f"len(configs): {len(configs)}")

submission = Runner(
    configs=configs,
    dataconfig=configs[args.fold],
    is_val=True,
    device=args.device,
).run(
    debug=args.debug,
    fold=args.fold,
)
print(submission)
submission.to_csv("submission.csv", index=False)

print(submission["step"].value_counts())
print(submission["event"].value_counts())

score_per_sid = dict()
for sid in submission["series_id"].unique():
    sub_sid = submission[submission["series_id"] == sid]
    valid_sol_sid = df_valid_solution[df_valid_solution["series_id"] == sid]
    if not isinstance(valid_sol_sid, pd.DataFrame):
        sub_sid = sub_sid.to_frame()
    if not isinstance(valid_sol_sid, pd.DataFrame):
        valid_sol_sid = valid_sol_sid.to_frame()

    if sub_sid.empty or valid_sol_sid.empty:
        continue
    score_per_sid[sid] = metrics.event_detection_ap(
        valid_sol_sid,
        sub_sid,
    )

min_score_sid = min(score_per_sid, key=score_per_sid.get)  # type: ignore
max_score_sid = max(score_per_sid, key=score_per_sid.get)  # type: ignore
print(f"min score sid: {min_score_sid}, score: {score_per_sid[min_score_sid]}")
print(f"max score sid: {max_score_sid}, score: {score_per_sid[max_score_sid]}")

print("\n score per sid")
pprint.pprint(score_per_sid)

cv_score = metrics.event_detection_ap(
    df_valid_solution,
    submission,
)

print(f"\n CV score: {cv_score}")


######## Analysis ########
def _analysis(
    preds: pl.DataFrame,
    sids: Sequence[str],
    save_dir: pathlib.Path,
    events_path: pathlib.Path,
    data_dir: pathlib.Path,
    use_features: Sequence[str],
) -> None:
    events = _load_events(events_path)
    for sid in sids:
        features = _load_features(data_dir, sid, use_features=use_features)
        events_this_sid = _filter_events(events, sid)
        preds_this_sid = _filter_events(preds, sid)
        fig, _ = _plot_events(features, events_this_sid, preds_this_sid)
        _save_fig(fig, f"events_{sid}.png", save_dir)


def _load_events(events_path: pathlib.Path) -> pl.DataFrame:
    events_df = pl.read_csv(events_path, dtypes={"step": pl.Int64})
    events_df = events_df.drop_nulls()
    return events_df


def _load_features(
    data_dir: pathlib.Path, sid: str, use_features: Sequence[str]
) -> pl.DataFrame:
    features_dir = data_dir / sid
    features: dict[str, np.ndarray] = {}
    for f_path in features_dir.glob("*.npy"):
        if (feature_name := f_path.stem) in use_features:
            features[feature_name] = np.load(f_path)
    assert len(features) > 0
    return pl.DataFrame(features)


def _filter_events(events: pl.DataFrame, sid: str) -> pl.DataFrame:
    return events.filter(pl.col("series_id") == sid)


def _plot_events(
    features: pl.DataFrame,
    events: pl.DataFrame,
    preds: pl.DataFrame,
    figsize: tuple[int, int] = (20, 10),
) -> tuple[figure.Figure, np.ndarray]:
    """plot features and events labels and preds for analysis

    Args:
        features (pl.DataFrame): features

        events (pl.DataFrame): events labels.
                            following the format of train_events.csv
                                - row_id: int
                                - series_id: str
                                - night: int
                                - event: str
                                - step: int
                                - timestamp: str

        preds (pl.DataFrame): preds
                            following the format of submission.csv
                                - row_id: int
                                - series_id: str
                                - event: str
                                - step: int
                                - score: float

        figsize (tuple[int, int], optional): figsize. Defaults to (20, 10).

    Returns:
        tuple[figure.Figure, np.ndarray]: fig, ax
    """
    print(f"features: {features.shape}")
    print(features)
    fig, ax = plt.subplots(len(features.columns), 1, figsize=figsize)
    assert isinstance(ax, np.ndarray)
    assert isinstance(fig, figure.Figure)

    # -- Plot features
    for i, feature_name in enumerate(features.columns):
        ax[i].plot(features[feature_name])
        ax[i].set_title(feature_name)
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("value")
        ax[i].grid()
        ax[i].legend()

    # -- Plot events labels
    for event_name, step in zip(events["event"], events["step"]):
        color = "red" if event_name == "onset" else "blue"
        for i in range(len(features.columns)):
            ax[i].axvline(
                step, color=color, linestyle="--", label=event_name, alpha=0.5
            )

    # -- Plot preds
    for event_name, step, score in zip(preds["event"], preds["step"], preds["score"]):
        color = "yellow" if event_name == "onset" else "green"
        for i in range(len(features.columns)):
            ax[i].axvline(
                step,
                color=color,
                linestyle="--",
                label="_".join([event_name, "pred"]),
                alpha=0.5,
            )

    for x in ax:
        handles, legends = x.get_legend_handles_labels()
        used_labels = set()
        unique_handles_legends = []
        for handle, legend in zip(handles, legends):
            if legend not in used_labels:
                unique_handles_legends.append((handle, legend))
                used_labels.add(legend)
        x.legend(*zip(*unique_handles_legends))
    return fig, ax


def _save_fig(fig: figure.Figure, save_path: str, save_dir: pathlib.Path) -> None:
    save_path_ = save_dir / save_path
    save_path_.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path_)


sids = [
    # "038441c925bb",
    # "fe90110788d2",
    min_score_sid,
    max_score_sid,
]
for sid in sids:
    valid_sol_sid = df_valid_solution[df_valid_solution["series_id"] == sid]
    sub_sid = submission[submission["series_id"] == sid]
    print(
        f"""
        SID: {sid}

        ########### solution ############

        {valid_sol_sid}

        ########### submission ##########

        {sub_sid}
    """
    )

_analysis(
    preds=pl.DataFrame._from_pandas(submission),
    sids=sids,
    save_dir=pathlib.Path(f"./output/analysis/{config.name}"),
    events_path=config.train_events_path,
    data_dir=pathlib.Path("./input/processed"),
    use_features=["anglez", "enmo", "hour_cos", "hour_sin"],
)


print(f"\n CV score: {cv_score}")
