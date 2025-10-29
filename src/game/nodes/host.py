from typing import Any, Dict

from ..config import resolve_behavior_mode
from ..modes.agent import nodes as agent_nodes
from ..modes.shared import HostNodeContext
from ..modes.workflow import nodes as workflow_nodes
from ..state import GameState

_HOST_DELEGATES = {
    "workflow": workflow_nodes,
    "agent": agent_nodes,
}


def _behavior_mode_from_state(state: GameState) -> str:
    """Resolve the active behavior mode using configuration fallbacks."""

    return resolve_behavior_mode(state=state)


def host_setup(state: GameState) -> Dict[str, Any]:
    """Initialize the game by delegating to the mode-specific host implementation."""

    mode = _behavior_mode_from_state(state)
    try:
        delegate = _HOST_DELEGATES[mode]
    except KeyError as exc:  # pragma: no cover - guardrail for future modes
        raise ValueError(f"Unsupported behavior mode: {mode}") from exc
    return delegate.host_setup(HostNodeContext(state=state))


def host_stage_switch(state: GameState) -> Dict[str, Any]:
    """Advance the stage by delegating to the mode-specific host implementation."""

    mode = _behavior_mode_from_state(state)
    try:
        delegate = _HOST_DELEGATES[mode]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported behavior mode: {mode}") from exc
    return delegate.host_stage_switch(HostNodeContext(state=state))


def host_result(state: GameState) -> Dict[str, Any]:
    """Resolve the current round by delegating to the mode-specific host implementation."""

    mode = _behavior_mode_from_state(state)
    try:
        delegate = _HOST_DELEGATES[mode]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported behavior mode: {mode}") from exc
    return delegate.host_result(HostNodeContext(state=state))
