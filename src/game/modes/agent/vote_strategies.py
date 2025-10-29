"""Reusable vote strategy helpers for agent behaviors."""

from __future__ import annotations

from typing import Dict

from ...state import GameState, PlayerMindset, alive_players
from ..workflow.behaviors import WorkflowPlayerBehavior


def vote_eliminate_prime(
    *,
    state: GameState,
    player_id: str,
    mindset: PlayerMindset,
    **_: object,
) -> str:
    """Aggressive strategy that mirrors the legacy suspicion-driven vote."""

    return WorkflowPlayerBehavior._decide_player_vote(state, player_id, mindset)


def vote_align_with_consensus(
    *,
    state: GameState,
    player_id: str,
    mindset: PlayerMindset,
    **_: object,
) -> str:
    """Follow the current majority unless it targets the agent."""

    valid_votes = {
        voter: vote
        for voter, vote in state.get("current_votes", {}).items()
        if getattr(vote, "phase_id", None) == state.get("phase_id")
    }

    if valid_votes:
        tally: Dict[str, int] = {}
        for vote in valid_votes.values():
            tally[vote.target] = tally.get(vote.target, 0) + 1

        if tally:
            # Prefer the current majority while avoiding self-sabotage.
            top_target = max(tally.items(), key=lambda item: (item[1], item[0]))[0]
            if top_target != player_id:
                return top_target

    return WorkflowPlayerBehavior._decide_player_vote(state, player_id, mindset)


def vote_counter_accuser(
    *,
    state: GameState,
    player_id: str,
    mindset: PlayerMindset,
    **_: object,
) -> str:
    """Defensive strategy that retaliates against players currently voting for the agent."""

    valid_votes = {
        voter: vote
        for voter, vote in state.get("current_votes", {}).items()
        if getattr(vote, "phase_id", None) == state.get("phase_id")
    }

    attackers = [
        voter
        for voter, vote in valid_votes.items()
        if vote.target == player_id and voter != player_id
    ]
    if attackers:
        return attackers[0]

    fallback = WorkflowPlayerBehavior._decide_player_vote(state, player_id, mindset)
    if fallback == player_id:
        other_alive = [pid for pid in alive_players(state) if pid != player_id]
        return other_alive[0] if other_alive else player_id
    return fallback


__all__ = [
    "vote_align_with_consensus",
    "vote_counter_accuser",
    "vote_eliminate_prime",
]