"""
Vote decision support tools.

This module provides tools for making voting decisions
based on accumulated suspicions and game state.

Access Control:
- Players can only access their own private state
- Players can access public game state
"""

from typing import Any, Dict
from datetime import datetime

from src.game.state import GameState, Vote, PlayerMindset


def decide_vote_tool(
    state: GameState,
    player_id: str,
) -> Dict[str, Any]:
    """
    Decide vote target based on player's suspicions and beliefs.

    This tool makes voting decisions by:
    - Analyzing player's self-belief and confidence
    - Evaluating suspicions about other players
    - Selecting the most likely spy target

    Vote Decision Logic:
    1. Determine own role (use opposite if confidence < 50%)
    2. Calculate scores for other players based on suspicions
    3. Vote for player with the highest suspicion score

    Access Control:
    - Can only access player's own private state (player_private_states[player_id])
    - Can access public game state (alive players, etc.)

    Args:
        state: Current GameState
        player_id: ID of the player making the vote decision

    Returns:
        Dictionary with new vote added to current_votes
    """

    # Access control: validate player_id exists
    if player_id not in state.get("player_private_states", {}):
        raise ValueError(f"Invalid player_id: {player_id}")

    # Check if it's voting phase
    if state["game_phase"] != "voting":
        return {}

    # Get player's private state (allowed)
    player_private = state["player_private_states"][player_id]
    player_mindset = player_private.playerMindset

    # Get alive players from public state
    eliminated = set(state.get("eliminated_players", []))
    alive = [p for p in state["players"] if p not in eliminated]

    # Ensure player is alive
    if player_id not in alive:
        return {}

    # Determine own role: if confidence > 50%, use current role, otherwise use opposite
    my_self_belief = player_mindset.self_belief
    my_role = my_self_belief.role
    if my_self_belief.confidence < 0.5:
        # Use opposite role (uncertain about self)
        my_role = "spy" if my_role == "civilian" else "civilian"

    # Calculate scores for other players based on suspicions
    player_scores: Dict[str, float] = {}
    for other_player_id in alive:
        if other_player_id == player_id:
            continue

        score = 0.0
        suspicions = player_mindset.suspicions or {}
        if other_player_id in suspicions:
            suspicion = suspicions[other_player_id]
            if my_role == suspicion.role:
                # Positive score means we trust them (same role alignment)
                score = suspicion.confidence
            else:
                # Negative score means we distrust them (different role alignment)
                score = -suspicion.confidence
        player_scores[other_player_id] = score

    # Select vote target
    if player_scores:
        # Pick the lowest score (most distrust) to target suspected opponents
        voted_target = min(player_scores, key=player_scores.get)
    else:
        # Fallback if no suspicions
        other_alive = [p for p in alive if p != player_id]
        if other_alive:
            voted_target = other_alive[0]  # Vote for the first other alive player
        elif alive:  # Only self is alive
            voted_target = player_id
        else:  # Should not happen in a valid game state
            raise ValueError("No alive players to vote for.")

    # Create vote record
    ts = int(datetime.now().timestamp() * 1000)
    new_vote = Vote(target=voted_target, ts=ts, phase_id=state["phase_id"])

    # Return update dictionary
    return {"current_votes": {player_id: new_vote}}


def analyze_voting_patterns(
    state: GameState,
    player_id: str,
) -> Dict[str, Any]:
    """
    Analyze voting patterns across the game.

    This tool analyzes:
    - Historical voting behavior
    - Vote targeting patterns
    - Bandwagon detection

    Access Control:
    - Can only access public voting data
    - Cannot access private states

    Args:
        state: Current GameState
        player_id: ID of the player performing analysis

    Returns:
        Analysis results with voting pattern insights
    """

    # Access control: validate player_id
    if player_id not in state.get("player_private_states", {}):
        raise ValueError(f"Invalid player_id: {player_id}")

    # Get current votes (public data)
    current_votes = state.get("current_votes", {})

    # Analyze vote distribution
    vote_counts: Dict[str, int] = {}
    for voter_id, vote in current_votes.items():
        target = vote.target
        vote_counts[target] = vote_counts.get(target, 0) + 1

    # Find most voted player
    most_voted = None
    max_votes = 0
    if vote_counts:
        most_voted = max(vote_counts, key=vote_counts.get)
        max_votes = vote_counts[most_voted]

    analysis = {
        "total_votes": len(current_votes),
        "vote_distribution": vote_counts,
        "most_voted_player": most_voted,
        "most_voted_count": max_votes,
        "is_bandwagon": max_votes
        > len(current_votes) / 2,  # Simple bandwagon detection
    }

    return analysis
