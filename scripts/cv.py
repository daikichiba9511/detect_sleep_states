import argparse
import importlib
import pprint

from src.run import Runner
from src.utils import LoggingUtils, get_class_vars

logger = LoggingUtils.get_stream_logger(20)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="exp000")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--is_val", action="store_true")
args = parser.parse_args()

config = importlib.import_module(f"src.configs.{args.config}").Config
logger.info(f"fold: {args.fold}, debug: {args.debug}, is_val: {args.is_val}")
logger.info(f"\n{pprint.pformat(get_class_vars(config))}")

runner = Runner(config=config, dataconfig=config, is_val=args.is_val)
submission = runner.run(debug=args.debug, fold=args.fold)
print(submission)
submission.to_csv("submission.csv", index=False)
