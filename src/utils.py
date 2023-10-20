import os
import random
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Any, ClassVar

import numpy as np
import torch

logger = getLogger(__name__)


class LoggingUtils:
    format: ClassVar[
        str
    ] = "[%(levelname)s] %(asctime)s - %(pathname)s : %(lineno)d: %(funcName)s: %(message)s"

    @classmethod
    def get_stream_logger(cls, level: int = INFO) -> Logger:
        logger = getLogger()
        logger.setLevel(level)

        handler = StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(Formatter(cls.format))
        logger.addHandler(handler)
        return logger

    @classmethod
    def add_file_handler(cls, logger: Logger, filename: str, level: int = INFO) -> None:
        handler = FileHandler(filename=filename)
        handler.setLevel(level)
        handler.setFormatter(Formatter(cls.format))
        logger.addHandler(handler)


def get_called_time() -> str:
    """Get current time in JST (Japan Standard Time = UTC+9)"""
    now = datetime.utcnow() + timedelta(hours=9)
    return now.strftime("%Y%m%d%H%M%S")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.autograd.anomaly_mode.set_detect_anomaly(False)


def get_class_vars(cls_obj: object) -> dict[str, Any]:
    return {k: v for k, v in cls_obj.__dict__.items() if not k.startswith("__")}


def measure_fn(logger=logger.info):
    def _timer(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = fn(*args, **kwargs)
            end_time = time.time()
            logger(f"{fn.__name__} done in {end_time - start_time:.4f} s")
            return result

        return wrapper

    return _timer


@contextmanager
def timer(name, log_fn=logger.info):
    t0 = time.time()
    yield
    log_fn(f"[{name}] done in {time.time() - t0:.3f} s")
