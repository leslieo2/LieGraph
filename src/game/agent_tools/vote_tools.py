from typing import Dict, Optional
from langchain.tools import tool

from src.game.state import GameState, PlayerMindset, alive_players
from src.game.strategy.serialization import normalize_mindset, to_plain_dict


def vote_tools(
    state: GameState, mindset_overrides: Optional[Dict[str, PlayerMindset]] = None
):
    """
    Bind voting tools against the shared state.

    The optional mindset_overrides allows callers (e.g., llm_decide_vote) to provide
    freshly inferred player mindsets before the reducer persists them back into state.
    This keeps the heuristic scoring in the tools aligned with the LLM's most recent
    analysis and avoids voting on stale beliefs.
    """
    mindset_overrides = mindset_overrides or {}

    def _get_player_mindset(player_id: str):
        """
        Resolve the latest mindset for the player from overrides or shared state.
        Normalization keeps downstream logic consistent.
        """
        if player_id in mindset_overrides:
            return normalize_mindset(mindset_overrides[player_id])

        player_private_state = state.get("player_private_states", {}).get(player_id, {})
        player_mindset = player_private_state.get("playerMindset", {})
        return normalize_mindset(player_mindset)

    @tool(description="vote for the most suspicion")
    def decide_player_vote(player_id: str) -> str:
        """
        Simplified vote decision logic:
        1. Determine own role (use opposite if confidence < 50%)
        2. Calculate scores for other players based on suspicions
        3. Vote for the second most suspicious player to avoid obvious patterns
        """

        # Get player's mindset from state
        mindset_state = _get_player_mindset(player_id)
        alive = alive_players(state)

        # Determine own role: if confidence > 50%, use current role, otherwise use opposite
        my_self_belief = mindset_state.get("self_belief", {})
        my_role = my_self_belief.get("role", "civilian")
        if my_self_belief.get("confidence", 0.0) < 0.5:
            # Use opposite role
            my_role = "spy" if my_role == "civilian" else "civilian"

        suspicions = mindset_state.get("suspicions", {}) or {}
        player_scores: Dict[str, float] = {}
        for other_player_id in alive:
            if other_player_id == player_id:
                continue

            score = 0.0
            suspicion = suspicions.get(other_player_id)
            if suspicion:
                suspicion_data = to_plain_dict(suspicion, lambda: {})
                suspicion_role = suspicion_data.get("role", "civilian")
                suspicion_conf = suspicion_data.get("confidence", 0.0)
                if my_role == suspicion_role:
                    # Positive score means we trust them (same role alignment)
                    score = suspicion_conf
                else:
                    # Negative score means we distrust them (different role alignment)
                    score = -suspicion_conf
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

    @tool(description="vote for the second suspicion")
    def decide_player_vote_second_best(player_id: str) -> str:
        """
        Vote decision logic that targets the second most suspicious player:
        1. Determine own role (use opposite if confidence < 50%)
        2. Calculate scores for other players based on suspicions
        3. Vote for the second most suspicious player to avoid obvious patterns
        """

        # Get player's mindset from state
        mindset_state = _get_player_mindset(player_id)
        alive = alive_players(state)

        # Determine own role: if confidence > 50%, use current role, otherwise use opposite
        my_self_belief = mindset_state.get("self_belief", {})
        my_role = my_self_belief.get("role", "civilian")
        if my_self_belief.get("confidence", 0.0) < 0.5:
            # Use opposite role
            my_role = "spy" if my_role == "civilian" else "civilian"

        suspicions = mindset_state.get("suspicions", {}) or {}
        player_scores: Dict[str, float] = {}
        for other_player_id in alive:
            if other_player_id == player_id:
                continue

            score = 0.0
            suspicion = suspicions.get(other_player_id)
            if suspicion:
                suspicion_data = to_plain_dict(suspicion, lambda: {})
                suspicion_role = suspicion_data.get("role", "civilian")
                suspicion_conf = suspicion_data.get("confidence", 0.0)
                if my_role == suspicion_role:
                    # Positive score means we trust them (same role alignment)
                    score = suspicion_conf
                else:
                    # Negative score means we distrust them (different role alignment)
                    score = -suspicion_conf
            player_scores[other_player_id] = score

        if player_scores:
            # Pick the second-lowest score (second most distrust) to avoid obvious voting patterns
            sorted_targets = sorted(player_scores, key=player_scores.get)
            if len(sorted_targets) >= 2:
                voted_target = sorted_targets[1]  # Second most suspicious
            else:
                voted_target = sorted_targets[0]  # Only one target available
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

    return [decide_player_vote, decide_player_vote_second_best]
