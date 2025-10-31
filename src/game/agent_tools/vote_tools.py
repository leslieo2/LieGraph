from typing import Dict

from src.game.state import GameState, alive_players, get_player_context
from src.game.strategy.serialization import normalize_mindset, to_plain_dict


def vote_tools(state: GameState):
    def decide_player_vote(state: GameState, player_id: str) -> str:
        """
        Simplified vote decision logic:
        1. Determine own role (use opposite if confidence < 50%)
        2. Calculate scores for other players based on suspicions
        3. Vote for player with the highest score
        """

        mindset_state = normalize_mindset(updated_mindset)
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
