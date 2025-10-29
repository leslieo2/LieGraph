"""Mode-specific behavior packages for LieGraph."""

from .registry import (
    BehaviorRegistry,
    create_behavior_registry,
    get_host_behavior,
    get_player_behavior,
    register_behavior_registry,
)
from .shared import (
    BehaviorResult,
    HostBehavior,
    HostNodeContext,
    PlayerBehavior,
    PlayerNodeContext,
)

__all__ = [
    "BehaviorRegistry",
    "BehaviorResult",
    "HostBehavior",
    "HostNodeContext",
    "PlayerBehavior",
    "PlayerNodeContext",
    "create_behavior_registry",
    "get_host_behavior",
    "get_player_behavior",
    "register_behavior_registry",
]
