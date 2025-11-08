"""
Centralized logging utilities for the LieGraph game engine.

Ensures every module uses a consistent logger configuration, while
still allowing runtime control via the ``LIEGRAPH_LOG_LEVEL`` env var.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_IS_CONFIGURED = False


def _configure_logging() -> None:
    """Configure the standard logging module once."""
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        return

    level_name = os.getenv("LIEGRAPH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _IS_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger scoped to the provided name."""
    _configure_logging()
    return logging.getLogger(name or "liegraph")
