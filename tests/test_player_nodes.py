import pytest
from unittest.mock import Mock, patch
from src.game.nodes.player import player_speech, player_vote
from src.game.state import (
    GameState,
    PlayerPrivateState,
    PlayerMindset,
    SelfBelief,
    Suspicion,
)

pytestmark = pytest.mark.workflow


@pytest.fixture
def player_id():
    return "a"


@pytest.fixture
def base_player_state(player_id):
    """Provides a base game state for player node tests."""
    return {
        "players": ["a", "b", "c", "d"],
        "game_id": "test_game",
        "current_round": 1,
        "game_phase": "speaking",
        "phase_id": "1:speaking:test",
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
                    self_belief=SelfBelief(role="civilian", confidence=0.5),
                    suspicions={},
                ),
            ),
            "b": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.5),
                    suspicions={},
                ),
            ),
            "c": PlayerPrivateState(
                assigned_word="orange",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="spy", confidence=0.5), suspicions={}
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


@patch("src.game.nodes.player.llm_generate_speech")
@patch("src.game.nodes.player.llm_update_player_mindset")
@patch("src.tools.llm.get_default_llm_client")
def test_player_speech(
    mock_get_client, mock_infer, mock_speech, player_id, base_player_state: GameState
):
    """Tests the player_speech node with mocked LLM calls."""
    # Arrange: Configure mocks to return predictable values
    mock_get_client.return_value = Mock()
    mock_infer.return_value = PlayerMindset(
        self_belief=SelfBelief(role="civilian", confidence=0.9),
        suspicions={"c": Suspicion(role="spy", confidence=0.7, reason="vague")},
    )
    mock_speech.return_value = "This is a test speech."

    # Act: Call the player_speech node
    update = player_speech(base_player_state, player_id)

    # Assert: Verify the output is correct
    assert "completed_speeches" in update
    speech = update["completed_speeches"][0]
    assert speech["player_id"] == player_id
    assert speech["content"] == "This is a test speech."

    assert "player_private_states" in update
    private_update = update["player_private_states"][player_id]
    assert private_update.playerMindset.self_belief.role == "civilian"
    assert private_update.playerMindset.suspicions["c"].role == "spy"

    # Verify mocks were called correctly
    mock_infer.assert_called_once()
    mock_speech.assert_called_once()


@patch("src.game.modes.workflow.behaviors.metrics_collector.on_vote_cast", create=True)
@patch("src.game.nodes.player.llm_update_player_mindset")
@patch("src.tools.llm.get_default_llm_client")
def test_player_vote(
    mock_get_client,
    mock_infer,
    mock_vote_metric,
    player_id,
    base_player_state: GameState,
):
    """Tests the player_vote node with mocked LLM calls."""
    # Arrange: Configure mocks
    mock_get_client.return_value = Mock()
    mock_infer.return_value = PlayerMindset(
        self_belief=SelfBelief(role="civilian", confidence=0.9),
        suspicions={"c": Suspicion(role="spy", confidence=0.8, reason="very vague")},
    )
    voting_state = base_player_state | {
        "game_phase": "voting",
        "phase_id": "1:voting:mock_uuid",
    }

    # Act: Call the player_vote node
    update = player_vote(voting_state, player_id)

    # Assert: Verify the output
    assert "current_votes" in update
    # With the current voting logic: civilian player suspects c as spy -> lowest score
    # Players b and d have neutral score (0.0), so a selects the most suspicious player "c"
    assert update["current_votes"][player_id].target == "c"

    assert "player_private_states" in update
    private_update = update["player_private_states"][player_id]
    assert private_update.playerMindset.self_belief.role == "civilian"

    # Verify mocks
    mock_infer.assert_called_once()


def test_player_speech_not_in_speaking_phase(base_player_state: GameState):
    """Tests that player_speech returns empty dict if not in speaking phase."""
    state = base_player_state | {"game_phase": "voting"}
    update = player_speech(state, "a")
    assert update == {}


def test_player_vote_not_in_voting_phase(base_player_state: GameState):
    """Tests that player_vote returns empty dict if not in voting phase."""
    state = base_player_state | {"game_phase": "speaking"}
    update = player_vote(state, "a")
    assert update == {}


def test_player_node_for_eliminated_player(base_player_state: GameState):
    """Tests that nodes do nothing for an eliminated player."""
    state = base_player_state | {"eliminated_players": ["a"]}
    speech_update = player_speech(state, "a")
    vote_update = player_vote(state | {"game_phase": "voting"}, "a")
    assert speech_update == {}
    assert vote_update == {}
