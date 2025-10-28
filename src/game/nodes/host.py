from typing import Any, Dict

from ..agents import HostNodeContext, get_host_behavior
from ..state import GameState

_DEFAULT_BEHAVIOR_MODE = "workflow"


def _behavior_mode_from_state(state: GameState) -> str:
    """Read the behavior mode from state, defaulting to workflow."""

    mode = state.get("behavior_mode")
    if isinstance(mode, str) and mode:
        return mode
    return _DEFAULT_BEHAVIOR_MODE


def host_setup(state: GameState) -> Dict[str, Any]:
    """Initializes the game by delegating to the configured host behavior."""

    behavior = get_host_behavior(mode=_behavior_mode_from_state(state))
    return behavior.setup(HostNodeContext(state=state))


def host_stage_switch(state: GameState) -> Dict[str, Any]:
    """Advances the stage by delegating to the configured host behavior."""

    behavior = get_host_behavior(mode=_behavior_mode_from_state(state))
    return behavior.stage_switch(HostNodeContext(state=state))


def host_result(state: GameState) -> Dict[str, Any]:
    """Resolves the current round by delegating to the configured host behavior."""

    behavior = get_host_behavior(mode=_behavior_mode_from_state(state))
    return behavior.resolve_round(HostNodeContext(state=state))
