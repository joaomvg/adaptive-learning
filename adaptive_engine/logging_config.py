import logging
import sys
from typing import Optional

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def configure_logging(
    level: int = logging.INFO,
) -> None:
    """
    Configure root logging once for the entire application.
    Safe to call multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with standard configuration.
    Ensures logging is configured exactly once.
    """
    configure_logging()
    return logging.getLogger(name)