"""Tests for agent-mode host and player behaviors."""

from typing import Any, Dict
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.agent

from src.game.modes.agent.behaviors import AgentPlayerBehavior, AgentHostBehavior
from src.game.modes.agent.toolbox import AgentToolbox
from src.game.modes.shared.interfaces import HostNodeContext, PlayerNodeContext
from src.game.config import get_config
from src.game.state import (
    GameState,
    PlayerMindset,
    PlayerPrivateState,
    SelfBelief,
    Suspicion,
)


@pytest.fixture
def agent_toolbox():
    """Provide a deterministic toolbox for tests."""

    def fake_mindset_updater(**kwargs: Any) -> PlayerMindset:
        suspicions = {
            "c": Suspicion(role="spy", confidence=0.85, reason="Odd clue"),
        }
        return PlayerMindset(
            self_belief=SelfBelief(role="civilian", confidence=0.75),
            suspicions=suspicions,
        )

    def fake_speech_generator(**kwargs: Any) -> str:
        strategy = kwargs.get("strategy", "none")
        return f"strategy:{strategy}"

    def fake_vote_selector(*, mindset: PlayerMindset, **_: Any) -> str:
        return next(iter(mindset.suspicions))

    return AgentToolbox(
        mindset_updater=fake_mindset_updater,
        speech_generator=fake_speech_generator,
        vote_selector=fake_vote_selector,
    )


@pytest.fixture
def base_state() -> Dict[str, Any]:
    config = get_config()
    players = ["a", "b", "c", "d"]
    private_states = {
        pid: PlayerPrivateState(
            assigned_word="apple",
            playerMindset=PlayerMindset(
                self_belief=SelfBelief(role="civilian", confidence=0.5),
                suspicions={},
            ),
        )
        for pid in players
    }
    private_states["c"] = PlayerPrivateState(
        assigned_word="orange",
        playerMindset=PlayerMindset(
            self_belief=SelfBelief(role="spy", confidence=0.6),
            suspicions={},
        ),
    )

    return {
        "game_id": "test-game",
        "players": players,
        "current_round": 1,
        "game_phase": "speaking",
        "phase_id": "1:speaking:test",
        "completed_speeches": [],
        "eliminated_players": [],
        "current_votes": {},
        "winner": None,
        "host_private_state": {
            "player_roles": {
                pid: ("spy" if pid == "c" else "civilian") for pid in players
            },
            "civilian_word": "apple",
            "spy_word": "orange",
        },
        "player_private_states": private_states,
        "behavior_mode": config.behavior_mode,
    }


def _speech_context(state: GameState, player_id: str) -> PlayerNodeContext:
    return PlayerNodeContext(
        state=state,
        player_id=player_id,
        extras={},
    )


def test_agent_player_behavior_speech(agent_toolbox, base_state):
    behavior = AgentPlayerBehavior(toolbox=agent_toolbox, llm_client=Mock())
    ctx = _speech_context(base_state, "a")

    update = behavior.decide_speech(ctx)

    assert "completed_speeches" in update
    speech = update["completed_speeches"][0]
    assert speech["player_id"] == "a"
    assert speech["content"].startswith("strategy:")

    memory = behavior.memory_for("a")
    assert memory.observations
    assert memory.decisions[-1].kind == "speech"
    assert memory.decisions[-1].strategy in {
        "seed-context",
        "press-lead",
        "reinforce",
        "blend-in",
    }


def test_agent_player_behavior_vote(agent_toolbox, base_state):
    behavior = AgentPlayerBehavior(toolbox=agent_toolbox, llm_client=Mock())
    voting_state = base_state | {"game_phase": "voting", "phase_id": "1:voting:test"}
    ctx = PlayerNodeContext(state=voting_state, player_id="a", extras={})

    update = behavior.decide_vote(ctx)

    assert "current_votes" in update
    vote = update["current_votes"]["a"]
    assert vote.target == "c"

    memory = behavior.memory_for("a")
    assert memory.decisions[-1].kind == "vote"
    assert memory.decisions[-1].target == "c"


def test_agent_host_behavior_journal(base_state):
    behavior = AgentHostBehavior()
    ctx = HostNodeContext(state=base_state)

    behavior.setup(ctx)
    base_state = base_state | {"game_phase": "speaking"}
    behavior.stage_switch(HostNodeContext(state=base_state))
    voting_state = base_state | {"game_phase": "voting"}
    behavior.resolve_round(HostNodeContext(state=voting_state))

    assert behavior.journal
    actions = [entry.action for entry in behavior.journal]
    assert "assign_roles" in actions
    assert "stage_switch" in actions
    assert "resolve_round" in actions
