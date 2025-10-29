"""Factory helpers for resolving host and player behavior implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

from .agent.behaviors import AgentHostBehavior, AgentPlayerBehavior
from .shared.interfaces import HostBehavior, PlayerBehavior
from .workflow.behaviors import WorkflowHostBehavior, WorkflowPlayerBehavior


@dataclass(slots=True)
class BehaviorRegistry:
    """Lightweight container for host and player behavior instances."""

    host_behavior: HostBehavior
    default_player_behavior: PlayerBehavior
    player_overrides: Dict[str, PlayerBehavior] = field(default_factory=dict)

    def host(self) -> HostBehavior:
        return self.host_behavior

    def for_player(self, player_id: str) -> PlayerBehavior:
        return self.player_overrides.get(player_id, self.default_player_behavior)


def create_behavior_registry(
    mode: str = "workflow",
    *,
    host_behavior: Optional[HostBehavior] = None,
    default_player_behavior: Optional[PlayerBehavior] = None,
    player_overrides: Optional[Mapping[str, PlayerBehavior]] = None,
) -> BehaviorRegistry:
    """Create a registry of behaviors according to the requested mode."""

    if mode == "workflow":
        host = host_behavior or WorkflowHostBehavior()
        default_player = default_player_behavior or WorkflowPlayerBehavior()
    elif mode == "agent":
        host = host_behavior or AgentHostBehavior()
        default_player = default_player_behavior or AgentPlayerBehavior()
    else:
        raise ValueError(f"Unsupported behavior mode: {mode}")

    return BehaviorRegistry(
        host_behavior=host,
        default_player_behavior=default_player,
        player_overrides=dict(player_overrides or {}),
    )


_REGISTRIES: Dict[str, BehaviorRegistry] = {
    "workflow": create_behavior_registry(mode="workflow"),
    "agent": create_behavior_registry(mode="agent"),
}


def register_behavior_registry(mode: str, registry: BehaviorRegistry) -> None:
    """Register or replace the registry for a specific mode."""

    _REGISTRIES[mode] = registry


def get_host_behavior(mode: str = "workflow") -> HostBehavior:
    """Retrieve the host behavior for the given mode."""

    try:
        return _REGISTRIES[mode].host()
    except KeyError as exc:  # pragma: no cover - guardrail for future modes
        raise ValueError(f"Unsupported behavior mode: {mode}") from exc


def get_player_behavior(player_id: str, mode: str = "workflow") -> PlayerBehavior:
    """Retrieve the behavior for the specified player."""

    try:
        registry = _REGISTRIES[mode]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported behavior mode: {mode}") from exc
    return registry.for_player(player_id)
