"""
Player node implementations for the "Who Is Spy" game.

This module contains the LangGraph nodes that handle player actions:
- Speech generation with strategic reasoning
- Voting decisions based on accumulated evidence
- Private state management and mindset updates

Each node represents a player's turn in the game flow and integrates
with the LLM strategy system to provide intelligent agent behavior.

Node Functions:
- player_speech: Generates speech using LLM reasoning and updates mindset
- player_vote: Makes voting decisions based on accumulated suspicions
- Helper functions: Context retrieval and private state management

Integration Points:
- Uses LLM strategy module for intelligent reasoning
- Integrates with state management for proper state updates
- Follows LangGraph node patterns for workflow integration
"""

from datetime import datetime
from typing import Dict, Any

from src.tools.llm import create_llm
from ..config import get_config
from ..metrics import metrics_collector
from ..state import (
    GameState,
    alive_players,
    create_speech_record,
    Speech,
    PlayerPrivateState,
    PlayerMindset,
    get_player_context,
    merge_probs,
)
from ..strategy import (
    llm_update_player_mindset,
    llm_generate_speech,
    llm_decide_vote,
    plan_player_speech,
)
from ..strategy.serialization import normalize_mindset
from .helpers import (
    get_assigned_word,
    get_private_state,
    get_normalized_player_mindset,
)


def _get_llm_client():
    """Create and return an LLM client instance.

    This function provides lazy initialization of the LLM client,
    creating it only when needed and allowing for runtime configuration.
    """
    return create_llm()


def _get_player_context(state: GameState, player_id: str):
    """
    Retrieves player-specific context needed for LLM interactions,
    including their private player_context, assigned word
    """
    player_context = get_player_context(state, player_id)
    private_state = get_private_state(player_context)
    assigned_word = get_assigned_word(private_state)
    return player_context, private_state, assigned_word


def _create_player_private_state_delta(
    updated_mindset: PlayerMindset,
    existing_private_state: Any,
    my_word,
) -> PlayerPrivateState:
    """
    Creates a PlayerPrivateState object with proper type validation, preserving assigned_word
    and merging suspicions from the player's current mindset with the updated mindset.
    """
    existing_state = get_normalized_player_mindset(existing_private_state)
    mindset_state = normalize_mindset(updated_mindset)

    new_suspicions = merge_probs(
        existing_state.get("suspicions", {}), mindset_state.get("suspicions", {})
    )

    return {
        "assigned_word": my_word,
        "playerMindset": {
            "self_belief": mindset_state.get("self_belief", {}),
            "suspicions": new_suspicions,
        },
    }


def player_speech(state: GameState, player_id: str) -> Dict[str, Any]:
    """
    Player node for generating speech.
    Calls LLM to infer identity and generate speech.
    """
    if state["game_phase"] != "speaking":
        return {}

    # Ensure player is alive before proceeding with speech generation
    if player_id not in alive_players(state):
        return {}

    # Get player-specific context
    _, existing_private_state, my_word = _get_player_context(state, player_id)

    print(
        f"🎤 PLAYER SPEECH: {player_id} is generating speech for round {state['current_round']}"
    )
    print(f"   Assigned word: {my_word}")

    # Generate playerMindset using LLM
    config = get_config()
    existing_player_mindset = get_normalized_player_mindset(existing_private_state)

    llm_client = _get_llm_client()
    updated_mindset = llm_update_player_mindset(
        llm_client=llm_client,
        my_word=my_word,
        completed_speeches=state["completed_speeches"],
        players=state["players"],
        alive=alive_players(state),
        me=player_id,
        rules=config.get_game_rules(),
        existing_player_mindset=existing_player_mindset,
    )

    updated_mindset_state = normalize_mindset(updated_mindset)

    try:
        speech_plan = plan_player_speech(state, player_id, updated_mindset_state)
    except Exception as exc:
        print(
            f"⚠️ SPEECH PLAN TOOL failed for {player_id}: {exc}, falling back without plan."
        )
        speech_plan = None

    # Generate speech using LLM
    new_speech_text = llm_generate_speech(
        llm_client=llm_client,
        my_word=my_word,
        self_belief=updated_mindset_state.get("self_belief", {}),
        suspicions=updated_mindset_state.get("suspicions", {}),
        completed_speeches=state["completed_speeches"],
        me=player_id,
        alive=alive_players(state),
        current_round=state["current_round"],
        speech_plan=speech_plan,
    )

    print(f'🎤 PLAYER SPEECH: {player_id} says: "{new_speech_text}"')
    print(f"   Self belief: {updated_mindset_state.get('self_belief')}")
    print(f"   Suspicions: {updated_mindset_state.get('suspicions')}")

    # Prepare the state updates based on the generated speech and PlayerMindset
    speech_record: Speech = create_speech_record(state, player_id, new_speech_text)

    metrics_collector.on_player_mindset_update(
        game_id=state.get("game_id"),
        round_number=state["current_round"],
        phase=state["game_phase"],
        player_id=player_id,
        mindset=updated_mindset_state,
    )
    metrics_collector.on_speech(
        game_id=state.get("game_id"),
        round_number=state["current_round"],
        player_id=player_id,
        content=new_speech_text,
    )

    delta_private = _create_player_private_state_delta(
        updated_mindset_state, existing_private_state, my_word
    )

    return {
        "completed_speeches": [speech_record],
        "player_private_states": {player_id: delta_private},
    }


def player_vote(state: GameState, player_id: str) -> Dict[str, Any]:
    """
    Player node for casting a vote.
    Calls LLM to infer identity and decide vote target.
    """
    if state["game_phase"] != "voting":
        return {}

    # Ensure player is alive before proceeding with vote
    if player_id not in alive_players(state):
        return {}

    # Get player-specific context for voting
    _, existing_private_state, my_word = _get_player_context(state, player_id)

    print(
        f"🗳️  PLAYER VOTE: {player_id} is deciding vote for round {state['current_round']}"
    )
    print(f"   Assigned word: {my_word}")

    # Generate playerMindset using LLM
    config = get_config()
    existing_player_mindset = get_normalized_player_mindset(existing_private_state)

    llm_client = _get_llm_client()
    updated_mindset = llm_update_player_mindset(
        llm_client=llm_client,
        my_word=my_word,
        completed_speeches=state["completed_speeches"],
        players=state["players"],
        alive=alive_players(state),
        me=player_id,
        rules=config.get_game_rules(),
        existing_player_mindset=existing_player_mindset,
    )
    updated_mindset_state = normalize_mindset(updated_mindset)
    # Decide on a vote target using the LLM with bound voting tools
    voted_target = llm_decide_vote(
        llm_client=llm_client,
        state=state,
        me=player_id,
        my_word=my_word,
        current_mindset=updated_mindset_state,
    )

    print(f"🗳️  PLAYER VOTE: {player_id} votes for: {voted_target}")
    print(f"   Self belief: {updated_mindset_state.get('self_belief')}")
    print(f"   Suspicions: {updated_mindset_state.get('suspicions')}")

    # Prepare the state updates based on the decided vote and PlayerMindset
    ts = int(datetime.now().timestamp() * 1000)

    metrics_collector.on_player_mindset_update(
        game_id=state.get("game_id"),
        round_number=state["current_round"],
        phase=state["game_phase"],
        player_id=player_id,
        mindset=updated_mindset_state,
    )

    delta_private = _create_player_private_state_delta(
        updated_mindset_state, existing_private_state, my_word
    )
    new_vote = {
        player_id: {
            "target": voted_target,
            "ts": ts,
            "phase_id": state["phase_id"],
        }
    }

    return {
        "current_votes": new_vote,
        "player_private_states": {player_id: delta_private},
    }
