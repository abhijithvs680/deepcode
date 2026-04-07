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


def quiet_app_loggers_for_tui() -> None:
    """
    Detach stdout/stderr StreamHandlers from `app.*` loggers.

    Logging during a Textual full-screen session corrupts the layout on Windows
    consoles (log lines paint over the TUI, then the next redraw "fixes" it).
    """
    import logging
    import sys

    streams = (sys.stdout, sys.stderr)
    for name in list(logging.root.manager.loggerDict.keys()):
        if not str(name).startswith("app"):
            continue
        log = logging.getLogger(name)
        if not isinstance(log, logging.Logger):
            continue
        for h in list(log.handlers):
            if isinstance(h, logging.StreamHandler):
                stream = getattr(h, "stream", None)
                if stream in streams:
                    log.removeHandler(h)
        if not log.handlers:
            log.addHandler(logging.NullHandler())
            log.propagate = False
