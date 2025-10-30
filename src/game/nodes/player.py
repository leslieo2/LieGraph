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
    Vote,
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
    private_state = player_context["private"]
    my_word = private_state.assigned_word
    return player_context, my_word


def _create_player_private_state_delta(
    updated_mindset: PlayerMindset,
    player_context: Dict[str, Any],
    my_word,
) -> PlayerPrivateState:
    """
    Creates a PlayerPrivateState object with proper type validation, preserving assigned_word
    and merging suspicions from the player's current mindset with the updated mindset.
    """
    private_state = player_context["private"]
    existing_suspicions = (
        private_state.playerMindset.suspicions if private_state.playerMindset else {}
    )

    return PlayerPrivateState(
        assigned_word=my_word,
        playerMindset=PlayerMindset(
            self_belief=updated_mindset.self_belief,
            suspicions=merge_probs(existing_suspicions, updated_mindset.suspicions),
        ),
    )


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
    cur_player_context, my_word = _get_player_context(state, player_id)

    print(
        f"🎤 PLAYER SPEECH: {player_id} is generating speech for round {state['current_round']}"
    )
    print(f"   Assigned word: {my_word}")

    # Generate playerMindset using LLM
    config = get_config()
    private_state = cur_player_context["private"]
    existing_player_mindset = private_state.playerMindset

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

    # Generate speech using LLM
    new_speech_text = llm_generate_speech(
        llm_client=llm_client,
        my_word=my_word,
        self_belief=updated_mindset.self_belief,
        suspicions=updated_mindset.suspicions,
        completed_speeches=state["completed_speeches"],
        me=player_id,
        alive=alive_players(state),
        current_round=state["current_round"],
    )

    print(f'🎤 PLAYER SPEECH: {player_id} says: "{new_speech_text}"')
    print(f"   Self belief: {updated_mindset.self_belief}")
    print(f"   Suspicions: {updated_mindset.suspicions}")

    # Prepare the state updates based on the generated speech and PlayerMindset
    speech_record: Speech = create_speech_record(state, player_id, new_speech_text)

    metrics_collector.on_player_mindset_update(
        game_id=state.get("game_id"),
        round_number=state["current_round"],
        phase=state["game_phase"],
        player_id=player_id,
        mindset=updated_mindset,
    )
    metrics_collector.on_speech(
        game_id=state.get("game_id"),
        round_number=state["current_round"],
        player_id=player_id,
        content=new_speech_text,
    )

    delta_private = _create_player_private_state_delta(
        updated_mindset, cur_player_context, my_word
    )

    return {
        "completed_speeches": [speech_record],
        "player_private_states": {player_id: delta_private},
    }


def _decide_player_vote(
    state: GameState,
    player_id: str,
    updated_mindset: Dict[str, Any],
) -> str:
    """
    Simplified vote decision logic:
    1. Determine own role (use opposite if confidence < 50%)
    2. Calculate scores for other players based on suspicions
    3. Vote for player with the highest score
    """

    alive = alive_players(state)

    # Determine own role: if confidence > 50%, use current role, otherwise use opposite
    my_self_belief = updated_mindset.self_belief
    my_role = my_self_belief.role
    if my_self_belief.confidence < 0.5:
        # Use opposite role
        my_role = "spy" if my_role == "civilian" else "civilian"

    player_scores: Dict[str, float] = {}
    for other_player_id in alive:
        if other_player_id == player_id:
            continue

        score = 0.0
        if other_player_id in updated_mindset.suspicions:
            suspicion = updated_mindset.suspicions[other_player_id]
            if my_role == suspicion.role:
                # Positive score means we trust them (same role alignment)
                score = suspicion.confidence
            else:
                # Negative score means we distrust them (different role alignment)
                score = -suspicion.confidence
        player_scores[other_player_id] = score

    if player_scores:
        # Pick the lowest score (most distrust) to target suspected opponents
        voted_target = min(player_scores, key=player_scores.get)
    else:
        # Fallback if no other players to score (e.g., only self is alive)
        other_alive = [p for p in alive if p != player_id]
        if other_alive:
            voted_target = other_alive[0]  # Vote for the first other alive player
        elif alive:  # Only self is alive
            voted_target = player_id
        else:  # Should not happen in a valid game state
            raise ValueError("No alive players to vote for.")

    return voted_target


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
    cur_player_context, my_word = _get_player_context(state, player_id)

    print(
        f"🗳️  PLAYER VOTE: {player_id} is deciding vote for round {state['current_round']}"
    )
    print(f"   Assigned word: {my_word}")

    # Generate playerMindset using LLM
    config = get_config()
    private_state = cur_player_context["private"]
    existing_player_mindset = private_state.playerMindset

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
    # Decide the player's vote and infer PlayerMindset using LLM
    voted_target = _decide_player_vote(state, player_id, updated_mindset)

    print(f"🗳️  PLAYER VOTE: {player_id} votes for: {voted_target}")
    print(f"   Self belief: {updated_mindset.self_belief}")
    print(f"   Suspicions: {updated_mindset.suspicions}")

    # Prepare the state updates based on the decided vote and PlayerMindset
    ts = int(datetime.now().timestamp() * 1000)

    metrics_collector.on_player_mindset_update(
        game_id=state.get("game_id"),
        round_number=state["current_round"],
        phase=state["game_phase"],
        player_id=player_id,
        mindset=updated_mindset,
    )

    delta_private = _create_player_private_state_delta(
        updated_mindset, cur_player_context, my_word
    )
    new_vote = {player_id: Vote(target=voted_target, ts=ts, phase_id=state["phase_id"])}

    return {
        "current_votes": new_vote,
        "player_private_states": {player_id: delta_private},
    }
