"""Shared helper functions for LangGraph game nodes."""

from typing import Any, Dict, Mapping

from ..strategy.serialization import normalize_mindset


def get_private_state(container: Mapping[str, Any] | None) -> Any:
    """Return the player private state from a context-like mapping."""
    if not container:
        return {}
    private_state = container.get("private")
    return private_state or {}


def get_assigned_word(private_state: Any) -> str:
    """Extract the assigned word from a player private state structure."""
    assigned_word = getattr(private_state, "assigned_word", None)
    if assigned_word is None and isinstance(private_state, dict):
        assigned_word = private_state.get("assigned_word")
    return assigned_word or ""


def get_normalized_player_mindset(private_state: Any) -> Dict[str, Any]:
    """Return the normalized player mindset from a private state."""
    if hasattr(private_state, "playerMindset"):
        raw_mindset = private_state.playerMindset
    elif isinstance(private_state, dict):
        raw_mindset = private_state.get("playerMindset")
    else:
        raw_mindset = None
    return normalize_mindset(raw_mindset)
