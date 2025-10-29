"""
Configuration management for the LieGraph game.

This module provides centralized configuration management for the game,
including player settings, vocabulary, and game balancing rules.

Features:
- YAML-based configuration with sensible defaults
- Player count validation and spy count calculation
- Vocabulary management with word pairs
- Name pool management for consistent player naming
- Game rule configuration for LLM interactions

Configuration Sources:
- config.yaml file in project root (if exists)
- Built-in defaults for development and testing
- Runtime validation to ensure configuration integrity

Game Balancing:
- Automatic spy count calculation based on player count
- Vocabulary validation and selection
- Player name pool management
"""

import os
from typing import List, Tuple

import yaml


class GameConfig:
    """Configuration class for the LieGraph game."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return self._get_default_config()

    @staticmethod
    def _get_default_config() -> dict:
        """Get default configuration."""
        return {
            "game": {
                "behavior_mode": "workflow",
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
            }
        }

    @property
    def player_count(self) -> int:
        """Get the configured number of players."""
        count = self._config["game"]["player_count"]
        min_players = self._config["game"]["settings"]["min_players"]
        max_players = self._config["game"]["settings"]["max_players"]

        # Validate player count
        if count < min_players:
            raise ValueError(f"Player count must be at least {min_players}")
        if count > max_players:
            raise ValueError(f"Player count cannot exceed {max_players}")

        return count

    @property
    def behavior_mode(self) -> str:
        """Get the configured behavior mode (workflow or agent)."""
        mode = self._config["game"].get("behavior_mode", "workflow")
        if mode not in {"workflow", "agent"}:
            raise ValueError(
                f"Unsupported behavior mode '{mode}'. Expected 'workflow' or 'agent'."
            )
        return mode

    @property
    def vocabulary(self) -> List[Tuple[str, str]]:
        """Get the vocabulary list."""
        vocab_list = self._config["game"]["vocabulary"]
        # Convert to tuples for consistency
        return [tuple(pair) for pair in vocab_list]

    @property
    def player_names_pool(self) -> List[str]:
        """Get the pool of available player names."""
        return self._config["game"]["player_names"]

    @property
    def max_rounds(self) -> int:
        """Get the maximum number of rounds per game."""
        return self._config["game"]["settings"]["max_rounds"]

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

        Returns:
            List of unique player names
        """
        count = self.player_count
        available_names = self.player_names_pool.copy()

        if count > len(available_names):
            raise ValueError(
                f"Cannot generate {count} unique names from pool of {len(available_names)} names"
            )

        # Select names in order for consistent results
        selected_names = available_names[:count]
        return selected_names

    def validate_config(self) -> bool:
        """Validate the configuration."""
        try:
            # Test player count validation
            _ = self.player_count

            # Test vocabulary
            vocab = self.vocabulary
            if not vocab:
                raise ValueError("Vocabulary list cannot be empty")

            # Test name generation
            names = self.generate_player_names()
            if len(names) != self.player_count:
                raise ValueError("Name generation failed")

            # Validate behavior mode
            _ = self.behavior_mode

            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
_config_instance: GameConfig = None


def get_config(config_path: str = None) -> GameConfig:
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
            # Try to find config.yaml in project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_root, "config.yaml")

        _config_instance = GameConfig(config_path)

    return _config_instance


def reload_config(config_path: str = None) -> GameConfig:
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
