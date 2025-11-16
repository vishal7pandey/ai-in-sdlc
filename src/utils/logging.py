"""Shared logging utilities for agents and services."""

from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with project defaults."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **context: Any,
) -> None:
    """Log a message enriched with structured context metadata."""

    level_name = level.upper()
    if not hasattr(logging, level_name):
        level_name = "INFO"
    serialized_context = json.dumps(context, default=str)
    logger.log(getattr(logging, level_name), f"{message} | context={serialized_context}")


__all__ = ["get_logger", "log_with_context"]
