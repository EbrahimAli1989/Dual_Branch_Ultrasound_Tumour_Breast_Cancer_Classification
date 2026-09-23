"""
Logging setup: console + optional file handler with unified format.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "us_clf",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure and return the root (or named) logger.

    Args:
        name     : logger name (use root logger name "" for all modules)
        log_file : optional path to a .txt log file
        level    : logging level (default INFO)
        fmt      : log format string
        datefmt  : date format string

    Returns:
        configured Logger instance
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    # Avoid adding duplicate handlers on re-entry
    if log.handlers:
        return log

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # File handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    return log
