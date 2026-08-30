import logging
import time
from contextlib import contextmanager

TEST = 25
logging.addLevelName(TEST, "TEST")
logging.Logger.test = lambda self, msg, *args, **kw: self.log(TEST, msg, *args, **kw)

__all__ = ["TEST", "get_logger", "logging", "setup_logging", "timer"]


class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: "\033[37m",  # white
        logging.INFO: "\033[34m",  # blue
        TEST: "\033[32m",  # green
        logging.WARNING: "\033[31m",  # red
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self._RESET}"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timer(label: str):
    _log = logging.getLogger(__name__)
    t0 = time.perf_counter()
    yield
    _log.info("[%s] %.4fs", label, time.perf_counter() - t0)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        _ColorFormatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(level=level, handlers=[handler])
