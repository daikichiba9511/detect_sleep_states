from datetime import datetime, timedelta
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import ClassVar

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
