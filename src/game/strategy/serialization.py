"""Serialization helpers for converting between Pydantic models and plain dicts."""

from typing import Any, Callable, TypeVar, cast

from src.game.state import PlayerMindset

T = TypeVar("T")


def to_plain_dict(value: Any, default_factory: Callable[[], T]) -> T:
    """
    Convert Pydantic models, TypedDict-like objects, or mappings to plain dicts.

    Args:
        value: The object to convert.
        default_factory: Callable that returns a default value when ``value`` is None.

    Returns:
        Plain dict (or TypedDict-compatible) representation of the input.
    """
    if value is None:
        return default_factory()

    if hasattr(value, "model_dump"):
        return cast(T, value.model_dump())

    if isinstance(value, dict):
        return cast(T, value)

    return cast(T, dict(value))


def _default_mindset() -> PlayerMindset:
    return {
        "self_belief": {"role": "civilian", "confidence": 0.5},
        "suspicions": {},
    }


def normalize_mindset(mindset: Any) -> PlayerMindset:
    """Normalize mindset-like inputs to the shared PlayerMindset dict structure."""
    return to_plain_dict(mindset, _default_mindset)
