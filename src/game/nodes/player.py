"""Player node entry points that delegate to configured behaviors."""

from typing import Any, Dict

from ..agents import PlayerNodeContext, get_player_behavior
from ..llm_strategy import llm_generate_speech, llm_update_player_mindset
from ..state import GameState

__all__ = [
    "player_speech",
    "player_vote",
    "llm_generate_speech",
    "llm_update_player_mindset",
]

_DEFAULT_BEHAVIOR_MODE = "workflow"


def _behavior_mode_from_state(state: GameState) -> str:
    """Read the behavior mode from state, defaulting to workflow."""

    mode = state.get("behavior_mode")
    if isinstance(mode, str) and mode:
        return mode
    return _DEFAULT_BEHAVIOR_MODE


def _build_player_behavior_extras() -> Dict[str, Any]:
    """Expose overridable primitives required by workflow behaviors."""

    return {
        "llm_update_player_mindset": llm_update_player_mindset,
        "llm_generate_speech": llm_generate_speech,
    }


def player_speech(state: GameState, player_id: str) -> Dict[str, Any]:
    """Generate player speech by delegating to the configured behavior."""

    behavior = get_player_behavior(player_id, mode=_behavior_mode_from_state(state))
    ctx = PlayerNodeContext(
        state=state,
        player_id=player_id,
        extras=_build_player_behavior_extras(),
    )
    return behavior.decide_speech(ctx)


def player_vote(state: GameState, player_id: str) -> Dict[str, Any]:
    """Generate player vote updates by delegating to the configured behavior."""

    behavior = get_player_behavior(player_id, mode=_behavior_mode_from_state(state))
    ctx = PlayerNodeContext(
        state=state,
        player_id=player_id,
        extras=_build_player_behavior_extras(),
    )
    return behavior.decide_vote(ctx)
