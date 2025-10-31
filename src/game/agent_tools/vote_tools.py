from typing import Dict, Optional
from langchain.tools import tool

from src.game.state import GameState, PlayerMindset, alive_players
from src.game.strategy.serialization import normalize_mindset, to_plain_dict


def vote_tools(
    state: GameState,
    bound_player_id: str,
    mindset_overrides: Optional[Dict[str, PlayerMindset]] = None,
):
    """
    Bind voting tools against the shared state.

    The optional mindset_overrides allows callers (e.g., llm_decide_vote) to provide
    freshly inferred player mindsets before the reducer persists them back into state.
    This keeps the heuristic scoring in the tools aligned with the LLM's most recent
    analysis and avoids voting on stale beliefs.

    The returned tools are zero-argument and always operate on the bound player, so
    downstream LLMs cannot accidentally vote using another player's mindset.
    """
    mindset_overrides = mindset_overrides or {}

    def _resolve_mindset() -> PlayerMindset:
        """
        Resolve the latest mindset for the bound player from overrides or shared state.
        Normalization keeps downstream logic consistent.
        """
        if bound_player_id in mindset_overrides:
            return normalize_mindset(mindset_overrides[bound_player_id])

        player_private_state = state.get("player_private_states", {}).get(
            bound_player_id, {}
        )
        player_mindset = player_private_state.get("playerMindset", {})
        return normalize_mindset(player_mindset)

    def _score_players(mindset_state: PlayerMindset) -> Dict[str, float]:
        alive = alive_players(state)
        my_self_belief = mindset_state.get("self_belief", {})
        my_role = my_self_belief.get("role", "civilian")
        if my_self_belief.get("confidence", 0.0) < 0.5:
            my_role = "spy" if my_role == "civilian" else "civilian"

        suspicions = mindset_state.get("suspicions", {}) or {}
        player_scores: Dict[str, float] = {}
        for other_player_id in alive:
            if other_player_id == bound_player_id:
                continue

            score = 0.0
            suspicion = suspicions.get(other_player_id)
            if suspicion:
                suspicion_data = to_plain_dict(suspicion, lambda: {})
                suspicion_role = suspicion_data.get("role", "civilian")
                suspicion_conf = suspicion_data.get("confidence", 0.0)
                score = suspicion_conf if my_role == suspicion_role else -suspicion_conf
            player_scores[other_player_id] = score
        return player_scores

    @tool(description="vote for the most suspicion")
    def decide_player_vote() -> str:
        """
        Simplified vote decision logic (player id pre-bound).
        """

        mindset_state = _resolve_mindset()
        alive = alive_players(state)
        player_scores = _score_players(mindset_state)

        if player_scores:
            return min(player_scores, key=player_scores.get)

        other_alive = [p for p in alive if p != bound_player_id]
        if other_alive:
            return other_alive[0]
        if alive:
            return bound_player_id
        raise ValueError("No alive players to vote for.")

    @tool(description="vote for the second suspicion")
    def decide_player_vote_second_best() -> str:
        """
        Vote decision logic targeting the second most suspicious player (player id pre-bound).
        """

        mindset_state = _resolve_mindset()
        alive = alive_players(state)
        player_scores = _score_players(mindset_state)

        if player_scores:
            sorted_targets = sorted(player_scores, key=player_scores.get)
            return sorted_targets[1] if len(sorted_targets) >= 2 else sorted_targets[0]

        other_alive = [p for p in alive if p != bound_player_id]
        if other_alive:
            return other_alive[0]
        if alive:
            return bound_player_id
        raise ValueError("No alive players to vote for.")

    return [decide_player_vote, decide_player_vote_second_best]
