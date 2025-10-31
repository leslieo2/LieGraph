from typing import Dict

from src.game.agent_tools.speech_tools import speech_planning_tools
from src.game.strategy import plan_player_speech
from src.game.state import PlayerMindset


def _make_state(
    mindset: PlayerMindset,
) -> Dict:
    """Minimal GameState stub focusing on the fields the planning tool reads."""
    return {
        "game_id": "test-game",
        "players": ["alice", "bob", "carol"],
        "current_round": 1,
        "game_phase": "speaking",
        "phase_id": "1:speaking:test",
        "completed_speeches": [],
        "eliminated_players": [],
        "current_votes": {},
        "winner": None,
        "host_private_state": {
            "player_roles": {},
            "civilian_word": "citrus",
            "spy_word": "lemon",
        },
        "player_private_states": {
            "alice": {"assigned_word": "orange", "playerMindset": mindset}
        },
    }


def test_plan_speech_confident_spy_blend_in():
    mindset: PlayerMindset = {
        "self_belief": {"role": "spy", "confidence": 0.82},
        "suspicions": {"bob": {"role": "civilian", "confidence": 0.2, "reason": ""}},
    }
    state = _make_state(mindset)

    plan_tool = speech_planning_tools(state, "alice")[0]
    plan = plan_tool.func()

    assert plan["goal"]["label"] == "blend_in"
    assert plan["clarity"] == "low"  # spy with early round keeps clarity low
    assert plan["self_role_view"] == "spy"
    assert plan["self_confidence"] == mindset["self_belief"]["confidence"]


def test_plan_speech_press_primary_suspect():
    mindset: PlayerMindset = {
        "self_belief": {"role": "civilian", "confidence": 0.76},
        "suspicions": {
            "bob": {"role": "spy", "confidence": 0.82, "reason": "Contradicting clues"},
            "carol": {
                "role": "spy",
                "confidence": 0.5,
                "reason": "Repeats others",
            },
        },
    }
    state = _make_state(mindset)
    state["current_round"] = 3  # later round should increase clarity for civilians

    plan_tool = speech_planning_tools(state, "alice")[0]
    plan = plan_tool.func()

    assert plan["goal"]["label"] == "press_primary_suspect"
    assert plan["clarity"] == "high"
    assert plan["top_suspicions"][0]["player_id"] == "bob"
    assert plan["top_suspicions"][0]["confidence"] == 0.82


def test_plan_player_speech_helper_prefers_override():
    stored_mindset: PlayerMindset = {
        "self_belief": {"role": "civilian", "confidence": 0.3},
        "suspicions": {},
    }
    override_mindset: PlayerMindset = {
        "self_belief": {"role": "spy", "confidence": 0.75},
        "suspicions": {},
    }
    state = _make_state(stored_mindset)

    plan = plan_player_speech(state, "alice", override_mindset)

    assert plan["self_role_view"] == "spy"
    assert plan["goal"]["label"] == "blend_in"
