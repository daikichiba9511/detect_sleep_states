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
from src import utils as my_utils
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

dataconfig = importlib.import_module(f"src.configs.{args.config}").Config()

####### UPDATE PARAMS #######
dataconfig.fold = int(args.fold)
# dataconfig.train_series = my_utils.load_series(
#     pathlib.Path("./input/for_train/folded_series_ids_fold5_seed42.json"),
#     "train_series",
#     int(args.fold),
# )
dataconfig.slide_size = dataconfig.seq_len // 2
dataconfig.valid_series = my_utils.load_series(
    pathlib.Path("./input/for_train/folded_series_ids_fold5_seed42.json"),
    "valid_series",
    int(args.fold),
)

logger.info(f"\n{pprint.pformat(get_class_vars(dataconfig))}")

########## VALID ##########
# df_valid_series = pl.read_parquet(config.train_series_path).filter(
#     pl.col("fold") == args.fold
# )
logger.info(f"fold: {args.fold}, debug: {args.debug}")

df_valid_series = dataconfig.valid_series
valid_data_type = "org"
# valid_data_type = "v1119"
# valid_data_type = "v1130"
if valid_data_type == "org":
    train_events_path = dataconfig.data_dir / "train_events.csv"
elif valid_data_type == "v1119":
    train_events_path = pathlib.Path("./input/processed/train_events_v1119.cscv")
else:
    train_events_path = pathlib.Path("./input/for_train/train_events_v1130.csv")
logger.info(f"train_events_path: {train_events_path}")

df_valid_events = pl.read_csv(train_events_path, dtypes={"step": pl.Int64}).drop_nulls()
df_valid_solution = (
    df_valid_events.filter(pl.col("series_id").is_in(df_valid_series))
    .to_pandas(use_pyarrow_extension_array=True)
    .dropna()
)
df_valid_solution = df_valid_solution[~df_valid_solution["step"].isnull()]

logger.info(f"df_valid_solution:\n{df_valid_solution}")


def _init_modelconfig(config_ver: str, fold: int, use_weight: str):
    config_ = importlib.import_module(f"src.configs.{config_ver}").Config()
    config_.spectrogram2dcnn_params["encoder_weights"] = None
    config_.slide_size = config_.seq_len // 2
    if use_weight == "full":
        config_.model_save_path = (
            config_.output_dir / f"full_{config_.name}_fold{fold}.pth"
        )
    elif use_weight == "last":
        config_.model_save_path = (
            config_.output_dir / f"last_{config_.name}_fold{fold}.pth"
        )
    elif use_weight == "best":
        config_.model_save_path = (
            config_.output_dir / f"{config_.name}_model_fold{fold}.pth"
        )
    else:
        raise ValueError(f"{use_weight=}")
    return config_


configs = []

# 個別のvalidation用
# for fold in range(args.fold + 1):
#     config = importlib.import_module(f"src.configs.{args.config}").Config
#     config_ = config()
#     config_.fold = args.fold
#
#     config_.model_save_path = (
#         # config_.output_dir / f"{config_.name}_model_fold{args.fold}.pth"
#         config_.output_dir / f"last_{config_.name}_fold{args.fold}.pth"
#     )
#     # config_.slide_size = config_.seq_len // 2
#     # logger.info(
#     #     "model_path: {model_path}".format(model_path=config_.model_save_path)
#     # )
#     configs.append(copy.deepcopy(config_))

# -- 083
# config_ = _init_modelconfig("exp083", 0, "last")
# # config_ = _init_modelconfig("exp083", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 084
# config_ = _init_modelconfig("exp084", 0, "last")
# # config_ = _init_modelconfig("exp083", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 085
# config_ = _init_modelconfig("exp085", 0, "last")
# config_ = _init_modelconfig("exp085", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 086
# config_ = _init_modelconfig("exp086", 0, "last")
# config_ = _init_modelconfig("exp086", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 087
# config_ = _init_modelconfig("exp087", 0, "last")
# config_ = _init_modelconfig("exp087", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 087_1
# config_ = _init_modelconfig("exp087_1", 0, "full")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 087_2
# config_ = _init_modelconfig("exp087_2", 0, "full")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 087_3
# config_ = _init_modelconfig("exp87_3", 0, "last")
# # config_ = _init_modelconfig("exp87_3", 1, "last")
# # config_ = _init_modelconfig("exp87_3", 2, "last")  # 0.0
# # config_ = _init_modelconfig("exp87_3", 3, "last")
# # config_ = _init_modelconfig("exp87_3", 4, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 088
# config_ = _init_modelconfig("exp088", 0, "last")
# config_ = _init_modelconfig("exp088", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 089
# config_ = _init_modelconfig("exp089", 0, "last")
# config_ = _init_modelconfig("exp089", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 090
# config_ = _init_modelconfig("exp090", 0, "last")
# config_ = _init_modelconfig("exp090", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 091
# config_ = _init_modelconfig("exp091", 0, "last")
# config_ = _init_modelconfig("exp091", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 092
# config_ = _init_modelconfig("exp092", 0, "last")
# # config_ = _init_modelconfig("exp092", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 093
# config_ = _init_modelconfig("exp093", 0, "last")
# config_ = _init_modelconfig("exp093", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 094
# config_ = _init_modelconfig("exp094", 0, "last")
# config_ = _init_modelconfig("exp094", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 098
# config_ = _init_modelconfig("exp098", 0, "last")
# # config_ = _init_modelconfig("exp094", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 100
# config_ = _init_modelconfig("exp100", 0, "last")
# # config_ = _init_modelconfig("exp100", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 101
# config_ = _init_modelconfig("exp101", 0, "last")
# # # config_ = _init_modelconfig("exp101", 1, "last")
# pprint.pprint(get_class_vars(config_))
# configs.append(copy.deepcopy(config_))

# -- 102
config_ = _init_modelconfig("exp102", 0, "last")
# # config_ = _init_modelconfig("exp102", 1, "last")
pprint.pprint(get_class_vars(config_))
configs.append(copy.deepcopy(config_))
print(f"len(configs): {len(configs)}")


submission = Runner(
    # configs=[configs[args.fold]],
    # configs=[configs[0], configs[2], configs[3], configs[4]],
    configs=[
        # configs[2],
        # configs[1],
        *configs
    ],
    dataconfig=dataconfig,
    is_val=True,
    device=args.device,
    valid_data_type=valid_data_type,
).run(debug=args.debug, fold=args.fold, score_thr=0.02, distance=90)


########## 周期的な部分の予測の除外 ##########
logger.info("remove periodic")
valid_series = (
    pl.read_parquet(dataconfig.data_dir / "train_series.parquet")
    .filter(pl.col("series_id").is_in(dataconfig.valid_series))
    .to_pandas(use_pyarrow_extension_array=True)
)
valid_periodic_dict = my_utils.create_periodic_dict(valid_series)
submission = my_utils.remove_periodic(submission, valid_periodic_dict)

########## NMS ##########
# logger.info("NMS")
# submission = my_utils.non_maximum_suppression(submission, iou_threshold=100)

print(submission)
submission.to_csv("submission.csv", index=False)

print(submission["step"].value_counts())
print(submission["event"].value_counts())


score_per_sid: dict[str, float] = dict()
for sid in submission["series_id"].unique():
    sub_sid = submission[submission["series_id"] == sid]
    valid_sol_sid = df_valid_solution[df_valid_solution["series_id"] == sid]

    if not isinstance(valid_sol_sid, pd.DataFrame):
        sub_sid = sub_sid.to_frame()
    if not isinstance(valid_sol_sid, pd.DataFrame):
        valid_sol_sid = valid_sol_sid.to_frame()

    if sub_sid.empty or valid_sol_sid.empty:
        logger.info(f"empty: {sid}, {sub_sid.shape=}, {valid_sol_sid.shape=}")
        continue

    score_per_sid[sid] = metrics.event_detection_ap(
        valid_sol_sid,
        sub_sid,
    )

min_score_sid_ = min(score_per_sid, key=score_per_sid.get)  # type: ignore
min_score_sid = (min_score_sid_, score_per_sid[min_score_sid_])
max_score_sid_ = max(score_per_sid, key=score_per_sid.get)  # type: ignore
max_score_sid = (max_score_sid_, score_per_sid[max_score_sid_])
print(f"min score sid: {min_score_sid}, score: {score_per_sid[min_score_sid_]}")
print(f"max score sid: {max_score_sid}, score: {score_per_sid[max_score_sid_]}")
less_than_07_sid = [(sid, score) for sid, score in score_per_sid.items() if score < 0.7]

print("\n score per sid")
pprint.pprint(score_per_sid)

cv_score = metrics.event_detection_ap(
    df_valid_solution,
    submission,
)

print(f"\n CV score: {cv_score}, ({dataconfig.name=})")


######## Analysis ########
def _analysis(
    preds: pl.DataFrame,
    sids: Sequence[tuple[str, float]],
    save_dir: pathlib.Path,
    events_path: pathlib.Path,
    data_dir: pathlib.Path,
    use_features: Sequence[str],
    fold: int = 0,
) -> None:
    events = _load_events(events_path)
    for sid, score in sids:
        features = _load_features(data_dir, sid, use_features=use_features)
        events_this_sid = _filter_events(events, sid)
        preds_this_sid = _filter_events(preds, sid)
        fig, _ = _plot_events(features, events_this_sid, preds_this_sid, score)
        _save_fig(fig, f"fold{fold}_events_{sid}.png", save_dir)
        plt.close("all")


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
    score: float,
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
    fig, ax = plt.subplots(len(features.columns) + 2, 1, figsize=figsize)
    assert isinstance(ax, np.ndarray)
    assert isinstance(fig, figure.Figure)

    # -- Plot features
    for i, feature_name in enumerate(features.columns):
        ax[i].plot(features[feature_name])
        ax[i].set_title(feature_name)
        ax[i].set_xlabel("step")
        ax[i].set_ylabel("value")
        ax[i].grid()
        ax[i].set_xlim(0, len(features))

    # -- Plot events labels
    for event_name, step in zip(events["event"], events["step"]):
        color = "red" if event_name == "onset" else "green"
        ax[len(features.columns)].axvline(
            step, color=color, linestyle="-", label=event_name, alpha=0.5
        )
        ax[len(features.columns)].set_xlim(0, len(features))
        ax[len(features.columns)].grid()

    # -- Plot preds
    for event_name, step, score in zip(preds["event"], preds["step"], preds["score"]):
        # color = "yellow" if event_name == "onset" else "aqua"
        color = "red" if event_name == "onset" else "green"
        ax[len(features.columns) + 1].axvline(
            step,
            ymin=0,
            ymax=score,
            color=color,
            linestyle="-",
            label="_".join([event_name, "pred"]),
            alpha=0.5,
        )
        ax[len(features.columns) + 1].set_xlim(0, len(features))
        ax[len(features.columns) + 1].grid()

    for x in ax[len(features.columns) :]:
        handles, legends = x.get_legend_handles_labels()
        used_labels = set()
        unique_handles_legends = []
        for handle, legend in zip(handles, legends):
            if legend not in used_labels:
                unique_handles_legends.append((handle, legend))
                used_labels.add(legend)
        x.legend(*zip(*unique_handles_legends))

    fig.suptitle(f"score: {score}")
    fig.tight_layout()
    return fig, ax


def _save_fig(fig: figure.Figure, save_path: str, save_dir: pathlib.Path) -> None:
    save_path_ = save_dir / save_path
    save_path_.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path_))


if cv_score > 0.6:
    sids: list[tuple[str, float]] = [
        # "038441c925bb",
        # "fe90110788d2",
        min_score_sid,
        max_score_sid,
        *less_than_07_sid,
    ]
    print(f"{min_score_sid=}, {max_score_sid=}, {less_than_07_sid=}")
    for sid, _ in sids:
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
        save_dir=pathlib.Path(f"./output/analysis/{dataconfig.name}"),
        events_path=dataconfig.train_events_path,
        data_dir=pathlib.Path("./input/processed"),
        use_features=["anglez", "enmo", "hour_cos", "hour_sin"],
        fold=int(args.fold),
    )


print(f"\n CV score: {cv_score}, {dataconfig.name=}, {args.fold=}")
