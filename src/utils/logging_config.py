"""ログ設定モジュール"""

import logging
import sys

from src.config import LOG_LEVEL


def setup_logging() -> logging.Logger:
    """アプリケーション全体のロガーを設定"""
    logger = logging.getLogger("swing")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
