"""
Speech planning tools for the player speech phase.

These tools give the LLM structured hooks to reason about tone, clarity,
and tactical focus before composing the actual utterance. Keeping the
planning logic in a tool makes it easy to unit test the heuristics while
reducing prompt complexity for the speech generation step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool

from src.game.state import GameState, PlayerMindset, alive_players
from src.game.strategy.builders.prompt_builder import determine_clarity
from src.game.strategy.serialization import normalize_mindset, to_plain_dict

SelfBeliefDict = Dict[str, Any]
SuspicionDict = Dict[str, Any]


def speech_planning_tools(
    state: GameState,
    bound_player_id: str,
    mindset_overrides: Optional[Dict[str, PlayerMindset]] = None,
) -> List[Any]:
    """
    Bind speech planning tools for a specific player.

    The optional ``mindset_overrides`` keeps planners aligned with the latest
    mindset even if the reducer has not yet persisted the update to shared
    state. This mirrors the pattern used in the voting tools.
    """

    mindset_overrides = mindset_overrides or {}

    def _resolve_mindset() -> PlayerMindset:
        """Return the freshest mindset for the bound player."""
        if bound_player_id in mindset_overrides:
            return normalize_mindset(mindset_overrides[bound_player_id])

        private_state = state.get("player_private_states", {}).get(bound_player_id, {})
        return normalize_mindset(private_state.get("playerMindset"))

    def _self_belief(mindset: PlayerMindset) -> SelfBeliefDict:
        return to_plain_dict(
            mindset.get("self_belief"),
            lambda: {"role": "civilian", "confidence": 0.5},
        )

    def _top_suspicions(
        mindset: PlayerMindset, top_k: int = 2
    ) -> List[Tuple[str, SuspicionDict]]:
        suspicions = to_plain_dict(mindset.get("suspicions"), dict)
        scored = []
        for player_id, suspicion in suspicions.items():
            suspicion_dict = to_plain_dict(
                suspicion,
                lambda: {"role": "civilian", "confidence": 0.0, "reason": ""},
            )
            scored.append((player_id, suspicion_dict))

        scored.sort(
            key=lambda item: float(item[1].get("confidence", 0.0)),
            reverse=True,
        )
        return scored[:top_k]

    def _select_goal(
        role: str, confidence: float, top_suspects: List[Tuple[str, SuspicionDict]]
    ) -> Dict[str, Any]:
        """
        Derive a coarse speech tactic based on role belief and suspicion spread.
        The goal label is intentionally simple so the LLM can elaborate.
        """
        if role == "spy" and confidence >= 0.6:
            return {
                "label": "blend_in",
                "reason": "High spy confidence — keep tone broad and non-committal.",
            }

        if top_suspects:
            top_conf = float(top_suspects[0][1].get("confidence", 0.0))
            second_conf = (
                float(top_suspects[1][1].get("confidence", 0.0))
                if len(top_suspects) > 1
                else 0.0
            )
            if top_conf >= 0.7 and (top_conf - second_conf) >= 0.15:
                return {
                    "label": "press_primary_suspect",
                    "reason": "One player is a clear outlier — add detail that separates them.",
                }
            if top_conf >= 0.55:
                return {
                    "label": "probe_suspects",
                    "reason": "Multiple players feel suspicious — craft hints that stress test them.",
                }

        if confidence >= 0.7:
            return {
                "label": "strengthen_allies",
                "reason": "Confident civilian — reinforce overlap to build trust.",
            }

        return {
            "label": "stay_neutral",
            "reason": "Uncertain alignment — keep ambiguity while gathering more signals.",
        }

    @tool(
        description="Plan the speech strategy (clarity, tone, focus) before speaking."
    )
    def plan_speech() -> Dict[str, Any]:
        """
        Return planning directives for the player's next speech.

        The plan includes clarity guidance, high-level goal, and suggested
        player targets to reference or avoid.
        """

        mindset = _resolve_mindset()
        belief = _self_belief(mindset)

        role = str(belief.get("role", "civilian"))
        confidence = float(belief.get("confidence", 0.5))

        alive = [pid for pid in alive_players(state) if pid != bound_player_id]
        current_round = int(state.get("current_round", 0))

        clarity_code, clarity_desc = determine_clarity(role, confidence, current_round)

        top_suspects = _top_suspicions(mindset)
        goal = _select_goal(role, confidence, top_suspects)

        suspects_payload = [
            {
                "player_id": player_id,
                "suspected_role": suspicion.get("role", "civilian"),
                "confidence": float(suspicion.get("confidence", 0.0)),
                "reason": suspicion.get("reason", ""),
            }
            for player_id, suspicion in top_suspects
        ]

        plan = {
            "player": bound_player_id,
            "round": current_round,
            "clarity": clarity_code,
            "clarity_reason": clarity_desc,
            "goal": goal,
            "self_role_view": role,
            "self_confidence": confidence,
            "alive_teammates": alive,
            "top_suspicions": suspects_payload,
        }

        print(
            "🛠️ SPEECH PLAN TOOL:",
            f"player={bound_player_id}",
            f"round={current_round}",
            f"clarity={clarity_code}",
            f"goal={goal.get('label')}",
        )

        return plan

    return [plan_speech]


__all__ = ["speech_planning_tools"]
