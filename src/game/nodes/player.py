"""Player node entry points that delegate to mode-specific behaviors."""

from typing import Any, Dict

from ..config import resolve_behavior_mode
from ..modes.agent import nodes as agent_nodes
from ..modes.shared import PlayerNodeContext
from ..modes.workflow import nodes as workflow_nodes
from ..state import GameState
from src.tools.llm.inference import llm_update_player_mindset
from src.tools.llm.speech import llm_generate_speech

__all__ = [
    "player_speech",
    "player_vote",
    "llm_generate_speech",
    "llm_update_player_mindset",
]

_PLAYER_DELEGATES = {
    "workflow": workflow_nodes,
    "agent": agent_nodes,
}


def _behavior_mode_from_state(state: GameState) -> str:
    """Resolve the active behavior mode using configuration fallbacks."""

    return resolve_behavior_mode(state=state)


def _build_player_behavior_extras() -> Dict[str, Any]:
    """Expose overridable primitives required by workflow behaviors."""

    return {
        "llm_update_player_mindset": llm_update_player_mindset,
        "llm_generate_speech": llm_generate_speech,
    }


def player_speech(state: GameState, player_id: str) -> Dict[str, Any]:
    """Generate player speech by delegating to the configured behavior."""

    mode = _behavior_mode_from_state(state)
    try:
        delegate = _PLAYER_DELEGATES[mode]
    except KeyError as exc:  # pragma: no cover - guardrail for future modes
        raise ValueError(f"Unsupported behavior mode: {mode}") from exc
    ctx = PlayerNodeContext(
        state=state,
        player_id=player_id,
        extras=_build_player_behavior_extras(),
    )
    return delegate.player_speech(ctx)


def player_vote(state: GameState, player_id: str) -> Dict[str, Any]:
    """Generate player vote updates by delegating to the configured behavior."""

    mode = _behavior_mode_from_state(state)
    try:
        delegate = _PLAYER_DELEGATES[mode]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported behavior mode: {mode}") from exc
    ctx = PlayerNodeContext(
        state=state,
        player_id=player_id,
        extras=_build_player_behavior_extras(),
    )
    return delegate.player_vote(ctx)
