import argparse
import importlib
import pprint

import pandas as pd
import polars as pl

from src import metrics
from src.run import Runner
from src.utils import LoggingUtils, get_class_vars
import copy

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


print(f"\n CV score: {cv_score}")
