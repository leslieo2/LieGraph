import pytest
from src.game.state import alive_players, next_alive_player
from src.game.rules import (
    assign_roles_and_words,
    calculate_eliminated_player,
    determine_winner,
)


@pytest.fixture
def players():
    return ["a", "b", "c", "d"]


def test_alive_players(players):
    state = {
        "players": players,
        "eliminated_players": ["b"],
    }
    assert alive_players(state) == ["a", "c", "d"]


def test_next_alive_player(players):
    state = {
        "players": players,
        "eliminated_players": ["d"],
        "current_round": 1,
        "completed_speeches": [
            {"round": 1, "player_id": "a", "content": "...", "ts": 0}
        ],
    }
    # a has spoken, b is next
    assert next_alive_player(state) == "b"

    state["completed_speeches"].append(
        {"round": 1, "player_id": "b", "content": "...", "ts": 1}
    )
    # a, b have spoken, c is next
    assert next_alive_player(state) == "c"

    state["completed_speeches"].append(
        {"round": 1, "player_id": "c", "content": "...", "ts": 2}
    )
    # a, b, c have spoken, no one is next
    assert next_alive_player(state) is None


def test_assign_roles(players):
    assignments = assign_roles_and_words(players)

    roles = assignments["host_private_state"]["player_roles"]
    spy_count = sum(1 for role in roles.values() if role == "spy")

    assert spy_count == 1
    assert len(assignments["player_private_states"]) == 4
    spy_word = assignments["host_private_state"]["spy_word"]
    civilian_word = assignments["host_private_state"]["civilian_word"]
    assert spy_word != civilian_word


def test_calculate_elimination():
    # Unique elimination
    state = {
        "current_votes": {
            "a": {"target": "c"},
            "b": {"target": "c"},
            "d": {"target": "a"},
        }
    }
    assert calculate_eliminated_player(state) == "c"

    # Tie vote -> randomly select one player from tied players
    state_tie = {
        "current_votes": {
            "a": {"target": "c"},
            "b": {"target": "d"},
            "c": {"target": "c"},
            "d": {"target": "d"},
        }
    }
    # In tie case, should randomly select one of the tied players
    eliminated = calculate_eliminated_player(state_tie)
    assert eliminated in ["c", "d"]

    # No votes
    state_no_votes = {"current_votes": {}}
    assert calculate_eliminated_player(state_no_votes) is None


def test_determine_winner(players):
    # 1 spy, 3 civilians
    host_state = {
        "player_roles": {"a": "civilian", "b": "spy", "c": "civilian", "d": "civilian"}
    }

    # No one eliminated yet, 1 spy vs 3 civilians -> game continues
    state = {"players": players, "eliminated_players": []}
    assert determine_winner(state, host_state) is None

    # One civilian eliminated, 1 spy vs 2 civilians -> game continues
    state["eliminated_players"] = ["a"]
    assert determine_winner(state, host_state) is None

    # Two civilians eliminated, 1 spy vs 1 civilian -> spies win
    state["eliminated_players"] = ["a", "c"]
    assert determine_winner(state, host_state) == "spies"

    # The spy is eliminated -> civilians win
    state["eliminated_players"] = ["b"]
    assert determine_winner(state, host_state) == "civilians"
