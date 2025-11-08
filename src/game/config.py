"""
Configuration management for the LieGraph game.

This module centralizes how configuration is loaded, validated, and exposed to
the rest of the codebase. It now layers a user-provided ``config.yaml`` on top
of built-in defaults and validates the merged result with Pydantic models so
configuration errors surface with clear messages at startup.

Configuration precedence:
1. Built-in defaults defined in ``DEFAULT_CONFIG``.
2. Values provided in ``config.yaml`` (or a custom path passed to ``get_config``),
   merged over the defaults.
3. Pydantic model defaults for any fields still unset after the merge.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from .logger import get_logger


class ConfigurationError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


DEFAULT_CONFIG: Dict[str, Any] = {
    "game": {
        "player_count": 4,
        "vocabulary": [
            ["苹果", "香蕉"],
            ["太阳", "月亮"],
            ["猫", "够=狗"],
            ["咖啡", "茶"],
            ["笔记本电脑", "书"],
        ],
        "player_names": [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
            "Frank",
            "Grace",
            "Henry",
            "Ivy",
            "Jack",
            "Katherine",
            "Leo",
            "Mia",
            "Noah",
            "Olivia",
            "Peter",
            "Quinn",
            "Rachel",
            "Sam",
            "Tina",
        ],
        "settings": {"min_players": 3, "max_players": 8, "max_rounds": 5},
    },
    "metrics": {"enabled": False},
}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""
    result: Dict[str, Any] = {}
    for key in base.keys() | overrides.keys():
        base_value = base.get(key)
        override_value = overrides.get(key)

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge(base_value, override_value)
        elif override_value is not None:
            result[key] = override_value
        else:
            result[key] = deepcopy(base_value)
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML data from the provided path."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse configuration file at {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration file at {path} must contain a top-level mapping."
        )

    return data


class GameSettingsModel(BaseModel):
    """Pydantic model for the game.settings section."""

    min_players: int = Field(default=3, ge=1)
    max_players: int = Field(default=8, ge=1)
    max_rounds: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_limits(self) -> "GameSettingsModel":
        if self.min_players > self.max_players:
            raise ValueError("min_players cannot exceed max_players")
        return self


class GameModel(BaseModel):
    """Pydantic model for the game section."""

    player_count: int = Field(default=4, ge=1)
    vocabulary: List[Tuple[str, str]] = Field(default_factory=list)
    player_names: List[str] = Field(default_factory=list)
    settings: GameSettingsModel = Field(default_factory=GameSettingsModel)

    @model_validator(mode="after")
    def validate_game(self) -> "GameModel":
        if not self.vocabulary:
            raise ValueError("Vocabulary list cannot be empty")
        for entry in self.vocabulary:
            if len(entry) != 2 or any(
                not isinstance(word, str) or not word for word in entry
            ):
                raise ValueError(
                    "Each vocabulary pair must contain two non-empty strings"
                )

        unique_names = set(self.player_names)
        if len(unique_names) != len(self.player_names):
            raise ValueError("Player names must be unique")
        if len(self.player_names) < self.player_count:
            raise ValueError(
                "Player name pool is smaller than the configured player count"
            )

        if (
            not self.settings.min_players
            <= self.player_count
            <= self.settings.max_players
        ):
            raise ValueError(
                "player_count must be between min_players and max_players (inclusive)"
            )
        return self


class MetricsConfigModel(BaseModel):
    """Configuration for optional metrics collection."""

    enabled: bool = False


class ProjectConfigModel(BaseModel):
    """Top-level Pydantic model for project configuration."""

    game: GameModel = Field(default_factory=GameModel)
    metrics: MetricsConfigModel = Field(default_factory=MetricsConfigModel)


class GameConfig:
    """Configuration class for the LieGraph game."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.config_path = Path(config_path).expanduser() if config_path else None
        self._config = self._load_config()

    def _load_config(self) -> ProjectConfigModel:
        """Load configuration from file, merge with defaults, and validate."""
        user_config: Dict[str, Any] = {}

        if self.config_path and self.config_path.exists():
            user_config = _load_yaml(self.config_path)

        merged = _deep_merge(deepcopy(DEFAULT_CONFIG), user_config)

        try:
            return ProjectConfigModel.model_validate(merged)
        except ValidationError as exc:
            detail = exc.errors()
            location = self.config_path or "built-in defaults"
            raise ConfigurationError(
                f"Invalid configuration in {location}: {detail}"
            ) from exc

    @property
    def player_count(self) -> int:
        """Get the configured number of players."""
        return self._config.game.player_count

    @property
    def vocabulary(self) -> List[Tuple[str, str]]:
        """Get the vocabulary list."""
        return [tuple(pair) for pair in self._config.game.vocabulary]

    @property
    def player_names_pool(self) -> List[str]:
        """Get the pool of available player names."""
        return list(self._config.game.player_names)

    @property
    def max_rounds(self) -> int:
        """Get the maximum number of rounds per game."""
        return self._config.game.settings.max_rounds

    @property
    def metrics_enabled(self) -> bool:
        """Return whether metrics collection is enabled."""
        return self._config.metrics.enabled

    def get_game_rules(self) -> dict:
        """
        Get game rules for LLM interactions.
        Returns a dict compatible with the old PUBLIC_RULES format.
        """
        return {
            "max_rounds": self.max_rounds,
            "spy_count": calculate_spy_count(self.player_count),
        }

    def generate_player_names(self) -> List[str]:
        """
        Generate player names based on configured player count.
        Returns consistent results by selecting names in order.
        """
        count = self.player_count
        available_names = self.player_names_pool

        if count > len(available_names):
            raise ValueError(
                f"Cannot generate {count} unique names from pool of {len(available_names)} names"
            )

        return available_names[:count]

    def validate_config(self) -> bool:
        """Validate the configuration."""
        try:
            _ = self.player_count
            _ = self.vocabulary
            names = self.generate_player_names()
            if len(names) != self.player_count:
                raise ValueError("Name generation failed")
            return True
        except Exception as exc:
            logger.error("Configuration validation failed: %s", exc)
            return False


# Global configuration instance
_config_instance: GameConfig | None = None
logger = get_logger(__name__)


def get_config(config_path: str | Path | None = None) -> GameConfig:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to configuration file. If None, uses default location.

    Returns:
        GameConfig instance
    """
    global _config_instance

    if _config_instance is None:
        if config_path is None:
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "config.yaml"

        _config_instance = GameConfig(config_path)

    return _config_instance


def reload_config(config_path: str | Path | None = None) -> GameConfig:
    """
    Reload the configuration from file.

    Args:
        config_path: Path to configuration file. If None, uses default location.

    Returns:
        GameConfig instance
    """
    global _config_instance
    _config_instance = None
    return get_config(config_path)


def calculate_spy_count(total_players: int) -> int:
    """
    Calculate the number of spies based on total players.
    Following common spy game balancing principles.
    """
    if total_players <= 4:
        return 1
    elif total_players <= 6:
        return 2
    elif total_players <= 8:
        return 2
    elif total_players <= 10:
        return 3
    else:
        return min(4, total_players // 3)  # Cap at 4 spies for very large games
