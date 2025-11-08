import pytest
from src.game.config import load_config
from src.game.metrics import GameMetrics
from src.game.nodes.host import host_setup, host_stage_switch, host_result


@pytest.fixture
def game_config():
    return load_config()


@pytest.fixture
def metrics():
    collector = GameMetrics()
    collector.set_enabled(False)
    return collector


@pytest.fixture
def base_state():
    """A base game state fixture for tests."""
    return {
        "players": ["a", "b", "c", "d", "e"],  # Changed to 5 players
        "game_id": "test_game",
        "current_round": 1,
        "game_phase": "setup",
        "completed_speeches": [],
        "eliminated_players": [],
        "current_votes": {},
        "host_private_state": {},
        "player_private_states": {},
    }


def test_host_setup(base_state, game_config, metrics):
    """Tests that host_setup initializes the game correctly."""
    update = host_setup(base_state, game_config=game_config, metrics=metrics)
    assert update["current_round"] == 1
    assert update["game_phase"] == "speaking"
    assert "host_private_state" in update
    assert "player_private_states" in update
    assert len(update["player_private_states"]) == 5


def test_host_stage_switch(base_state):
    """Tests the logic for switching from speaking to voting phase."""
    # Case 1: Speaking phase is not over
    speaking_state = base_state | {"game_phase": "speaking"}
    update = host_stage_switch(speaking_state)
    assert (
        update == {}
    )  # No state update needed - next_alive_player is computed dynamically

    # Case 2: Last player has spoken
    speaking_state_done = speaking_state | {
        "completed_speeches": [
            {"round": 1, "player_id": "a", "content": ""},
            {"round": 1, "player_id": "b", "content": ""},
            {"round": 1, "player_id": "c", "content": ""},
            {"round": 1, "player_id": "d", "content": ""},
            {"round": 1, "player_id": "e", "content": ""},  # Added speech for 'e'
        ]
    }
    update_done = host_stage_switch(speaking_state_done)
    assert update_done.get("game_phase") == "voting"
    assert "phase_id" in update_done


def test_host_result_elimination_and_advance(base_state, metrics):
    """Tests a standard round result: one player is eliminated and the game advances."""
    # Scenario: 5 players, 1 spy (b), 4 civilians (a,c,d,e)
    # Eliminate a civilian ('a'). Game should continue.
    voting_state = base_state | {
        "game_phase": "voting",
        "current_votes": {
            "b": {"target": "a"},  # Spy votes for 'a'
            "c": {"target": "a"},  # Civilian votes for 'a'
            "d": {"target": "b"},  # Civilian votes for 'b'
            "e": {"target": "a"},  # Civilian votes for 'a'
        },
        "host_private_state": {
            "player_roles": {
                "a": "civilian",
                "b": "spy",
                "c": "civilian",
                "d": "civilian",
                "e": "civilian",
            }
        },
    }
    update = host_result(voting_state, metrics=metrics)

    assert update["game_phase"] == "speaking"
    assert update["current_round"] == 2
    assert update["eliminated_players"] == ["a"]  # 'a' gets 3 votes
    assert update["current_votes"] == {}


def test_host_result_spy_win(base_state, metrics):
    """Tests the condition for a spy victory."""
    voting_state = base_state | {
        "game_phase": "voting",
        "players": ["a", "b", "c"],
        "eliminated_players": ["d"],
        "current_votes": {"a": {"target": "c"}},  # 'c' will be eliminated
        "host_private_state": {
            "player_roles": {
                "a": "civilian",
                "b": "spy",
                "c": "civilian",
                "d": "civilian",
            }
        },
    }
    # After 'c' is eliminated, 1 spy ('b') and 1 civilian ('a') will remain. Spies win.
    update = host_result(voting_state, metrics=metrics)
    assert update["game_phase"] == "result"
    assert update["eliminated_players"] == ["c"]
    assert update["winner"] == "spies"


def test_host_result_civilian_win(base_state, metrics):
    """Tests the condition for a civilian victory."""
    voting_state = base_state | {
        "game_phase": "voting",
        "players": ["a", "b"],
        "eliminated_players": ["c", "d"],
        "current_votes": {"a": {"target": "b"}},  # The spy 'b' is voted out
        "host_private_state": {
            "player_roles": {
                "a": "civilian",
                "b": "spy",
                "c": "civilian",
                "d": "civilian",
            }
        },
    }
    update = host_result(voting_state, metrics=metrics)
    assert update["game_phase"] == "result"
    assert update["eliminated_players"] == ["b"]
    assert update["winner"] == "civilians"
