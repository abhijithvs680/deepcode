"""
Structured logging configuration.

Usage:
    from app.core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Server started", extra={"port": 8000})
"""
import logging
import sys

from app.core.config import get_settings


def get_logger(name: str) -> logging.Logger:
    """
    Create a configured logger instance.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A logging.Logger with proper formatting and level.
    """
    settings = get_settings()

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger
