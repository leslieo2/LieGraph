"""Tests for workflow-aligned behavior implementations."""
from typing import Any
from unittest.mock import Mock

import pytest

from src.game.agents.interfaces import PlayerNodeContext
from src.game.agents.workflow_behaviors import WorkflowPlayerBehavior
from src.game.state import (
    PlayerMindset,
    PlayerPrivateState,
    SelfBelief,
    Suspicion,
)


@pytest.fixture
def player_behavior() -> WorkflowPlayerBehavior:
    return WorkflowPlayerBehavior()


@pytest.fixture
def base_state():
    return {
        "players": ["a", "b", "c", "d"],
        "game_id": "test_game",
        "current_round": 1,
        "phase_id": "1:speaking:test",
        "game_phase": "speaking",
        "completed_speeches": [],
        "eliminated_players": [],
        "current_votes": {},
        "winner": None,
        "host_private_state": {
            "player_roles": {
                "a": "civilian",
                "b": "civilian",
                "c": "spy",
                "d": "civilian",
            },
            "civilian_word": "apple",
            "spy_word": "orange",
        },
        "player_private_states": {
            "a": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.6),
                    suspicions={},
                ),
            ),
            "b": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.6),
                    suspicions={},
                ),
            ),
            "c": PlayerPrivateState(
                assigned_word="orange",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="spy", confidence=0.7),
                    suspicions={},
                ),
            ),
            "d": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.5),
                    suspicions={},
                ),
            ),
        },
    }


def _mindset_stub(**_: Any) -> PlayerMindset:
    return PlayerMindset(
        self_belief=SelfBelief(role="civilian", confidence=0.9),
        suspicions={
            "c": Suspicion(role="spy", confidence=0.8, reason="Imprecise clue"),
        },
    )


def test_workflow_behavior_decide_speech(player_behavior, base_state):
    update_mindset = Mock(side_effect=_mindset_stub)
    generate_speech = Mock(return_value="Cozy words about apples.")
    ctx = PlayerNodeContext(
        state=base_state,
        player_id="a",
        extras={
            "llm_update_player_mindset": update_mindset,
            "llm_generate_speech": generate_speech,
        },
    )

    result = player_behavior.decide_speech(ctx)

    assert "completed_speeches" in result
    speech_record = result["completed_speeches"][0]
    assert speech_record["player_id"] == "a"
    assert speech_record["content"] == "Cozy words about apples."

    assert "player_private_states" in result
    private_state = result["player_private_states"]["a"]
    assert private_state.playerMindset.self_belief.role == "civilian"
    assert private_state.playerMindset.suspicions["c"].role == "spy"

    update_mindset.assert_called_once()
    generate_speech.assert_called_once()


def test_workflow_behavior_decide_vote(player_behavior, base_state):
    base_state = base_state | {
        "game_phase": "voting",
        "phase_id": "1:voting:test",
    }
    update_mindset = Mock(side_effect=_mindset_stub)
    ctx = PlayerNodeContext(
        state=base_state,
        player_id="a",
        extras={
            "llm_update_player_mindset": update_mindset,
        },
    )

    result = player_behavior.decide_vote(ctx)

    assert "current_votes" in result
    vote = result["current_votes"]["a"]
    assert vote.target == "c"
    assert vote.phase_id == "1:voting:test"

    assert "player_private_states" in result
    private_state = result["player_private_states"]["a"]
    assert private_state.playerMindset.self_belief.role == "civilian"
    assert private_state.playerMindset.suspicions["c"].role == "spy"

    update_mindset.assert_called_once()
