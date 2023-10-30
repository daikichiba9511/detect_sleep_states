import argparse
import importlib
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.metrics import column_names, score, tolerances
from src.run import Runner
from src.utils import LoggingUtils, get_class_vars
from tqdm.auto import tqdm
import copy

logger = LoggingUtils.get_stream_logger(20)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="exp000")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--all", default=False, action="store_true")
args = parser.parse_args()

config = importlib.import_module(f"src.configs.{args.config}").Config

####### UPDATE PARAMS #######
config.fold = args.fold
config.model_save_path = config.output_dir / f"{config.name}_model_fold{args.fold}.pth"


logger.info(f"fold: {args.fold}, debug: {args.debug}")
logger.info(f"\n{pprint.pformat(get_class_vars(config))}")

########## VALID ##########
df_valid_series = pl.read_parquet(config.train_series_path).filter(
    pl.col("fold") == args.fold
)
df_valid_series = df_valid_series.cast({"step": pl.Int64})
df_valid_events = pl.read_csv(config.train_events_path, dtypes={"step": pl.Int64})
df_valid_solution = df_valid_events.filter(
    pl.col("series_id").is_in(df_valid_series["series_id"].unique())
).to_pandas(use_pyarrow_extension_array=True)
df_valid_solution = df_valid_solution[~df_valid_solution["step"].isnull()]
print(df_valid_solution)

configs = []
if args.all:
    for fold in range(5):
        config_ = copy.deepcopy(config)
        config_.fold = fold
        config_.model_save_path = (
            config_.output_dir / f"{config_.name}_model_fold{fold}.pth"
        )
        configs.append(config_)
else:
    configs.append(config)

submission = Runner(configs=configs, dataconfig=config, is_val=True, device="cpu").run(
    debug=args.debug, fold=args.fold
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
    score_per_sid[sid] = score(
        valid_sol_sid,
        sub_sid,
        tolerances,  # type: ignore
        **column_names,  # type: ignore
    )

min_score_sid = min(score_per_sid, key=score_per_sid.get)  # type: ignore
max_score_sid = max(score_per_sid, key=score_per_sid.get)  # type: ignore
print(f"min score sid: {min_score_sid}, score: {score_per_sid[min_score_sid]}")
print(f"max score sid: {max_score_sid}, score: {score_per_sid[max_score_sid]}")

print("\n score per sid")
pprint.pprint(score_per_sid)

cv_score = score(
    df_valid_solution,
    submission,
    tolerances,  # type: ignore
    **column_names,  # type: ignore
)

print(f"\n CV score: {cv_score}")

sids = [
    "038441c925bb",
    "fe90110788d2",
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

    df_valid_series_sid = df_valid_series.filter(pl.col("series_id") == sid).to_pandas(
        use_pyarrow_extension_array=True
    )
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    ax[0].plot(df_valid_series_sid["anglez"])
    ax[0].set_title("angle_z")
    ax[1].plot(df_valid_series_sid["enmo"])
    ax[1].set_title("enmo")
    # eventの起きたタイミングをプロット
    # red: label_onset
    # green: label_wakeup
    # yellow: pred_onset
    # purple: pred_wakeup
    for event_step, event_type in tqdm(
        zip(valid_sol_sid["step"].values, valid_sol_sid["event"].values),
        desc="Make Plot",
    ):
        color = "red" if event_type == "onset" else "green"
        ax[0].axvline(event_step, color=color, alpha=0.5, label="label_" + event_type)
        ax[1].axvline(event_step, color=color, alpha=0.5, label="label_" + event_type)

    for event_step, event_type in zip(sub_sid["step"].values, sub_sid["event"].values):
        color = "yellow" if event_type == "onset" else "purple"
        ax[0].axvline(event_step, color=color, alpha=0.5, label="pred_" + event_type)
        ax[1].axvline(event_step, color=color, alpha=0.5, label="pred_" + event_type)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    save_dir = config.output_dir.parent / "analysis"
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / f"{config.name}_sid-{sid}_fold{args.fold}.png")
    plt.close("all")
plt.close("all")
print(f"\n CV score: {cv_score}")
