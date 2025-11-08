"""
Lightweight dependency container for wiring runtime services into the game.

Instead of relying on module-level singletons (e.g., global config instances or
metrics collectors), we bundle the required collaborators into a simple data
class and pass them explicitly where needed. This makes it trivial to spin up
multiple, isolated games for tests or concurrent executions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import GameConfig, load_config
from .metrics import GameMetrics


@dataclass(slots=True)
class GameDependencies:
    """Container object that holds the runtime services a game needs."""

    config: GameConfig
    metrics: GameMetrics


def build_dependencies(
    *,
    config: GameConfig | None = None,
    metrics: GameMetrics | None = None,
    config_path: str | Path | None = None,
) -> GameDependencies:
    """
    Construct a ``GameDependencies`` instance.

    Args:
        config: Optional pre-built ``GameConfig``.
        metrics: Optional ``GameMetrics`` instance (useful for sharing collectors).
        config_path: Optional config path when ``config`` is not supplied.
    """
    cfg = config or load_config(config_path)
    collector = metrics or GameMetrics()
    return GameDependencies(config=cfg, metrics=collector)
