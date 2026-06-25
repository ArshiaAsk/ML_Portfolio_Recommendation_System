"""Centralized logging configuration for portfolio_ml."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
        level: Optional log level override (e.g. ``"DEBUG"``). Defaults to INFO.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    effective_level = level or "INFO"
    logger.setLevel(getattr(logging, effective_level.upper(), logging.INFO))
    logger.propagate = False

    return logger
