"""Behavior abstraction contracts for host and player nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from ...state import GameState

# Shared type alias for return payloads produced by behavior implementations.
BehaviorResult = Dict[str, Any]


@dataclass(slots=True)
class HostNodeContext:
    """Context object passed to host behavior methods."""

    state: GameState
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlayerNodeContext:
    """Context object passed to player behavior methods."""

    state: GameState
    player_id: str
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HostBehavior(Protocol):
    """Behavior contract for host-related LangGraph nodes."""

    def setup(self, ctx: HostNodeContext) -> BehaviorResult:
        """Prepare initial game state updates before the first round."""

    def stage_switch(self, ctx: HostNodeContext) -> BehaviorResult:
        """Select the next stage or speaker given the shared state."""

    def resolve_round(self, ctx: HostNodeContext) -> BehaviorResult:
        """Aggregate round results and determine eliminations or winners."""


@runtime_checkable
class PlayerBehavior(Protocol):
    """Behavior contract for per-player decision nodes."""

    def decide_speech(self, ctx: PlayerNodeContext) -> BehaviorResult:
        """Produce the speech update for the designated player."""

    def decide_vote(self, ctx: PlayerNodeContext) -> BehaviorResult:
        """Return the vote update for the designated player."""
