"""
Unit tests for identity inference tools.

Tests cover:
- update_player_mindset_tool functionality
- Access control validation
- State updates and merging
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.game.agents.tools.identity import (
    update_player_mindset_tool,
    analyze_speech_consistency,
)
from src.game.state import (
    GameState,
    PlayerPrivateState,
    PlayerMindset,
    SelfBelief,
    Suspicion,
    Speech,
)


@pytest.fixture
def mock_game_state():
    """Create a mock game state for testing."""
    return {
        "game_id": "test-game-1",
        "players": ["player1", "player2", "player3"],
        "current_round": 1,
        "game_phase": "speaking",
        "phase_id": "1:speaking:abc123",
        "completed_speeches": [
            {
                "round": 1,
                "seq": 0,
                "player_id": "player2",
                "content": "It's something you can eat",
                "ts": 1234567890,
            }
        ],
        "eliminated_players": [],
        "current_votes": {},
        "winner": None,
        "host_private_state": {
            "player_roles": {
                "player1": "spy",
                "player2": "civilian",
                "player3": "civilian",
            },
            "civilian_word": "apple",
            "spy_word": "banana",
        },
        "player_private_states": {
            "player1": PlayerPrivateState(
                assigned_word="banana",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.5),
                    suspicions={},
                ),
            ),
            "player2": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.7),
                    suspicions={},
                ),
            ),
            "player3": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.6),
                    suspicions={},
                ),
            ),
        },
    }


@pytest.fixture
def mock_runtime(mock_game_state):
    """Create a mock ToolRuntime."""
    # Not needed anymore since we pass state directly
    return mock_game_state


def test_update_player_mindset_tool_access_control(mock_runtime):
    """Test that invalid player_id raises ValueError."""
    with pytest.raises(ValueError, match="Invalid player_id"):
        update_player_mindset_tool(mock_runtime, "invalid_player")


@patch("src.tools.llm.create_llm")
@patch("src.game.agents.tools.identity.create_extractor")
def test_update_player_mindset_tool_success(mock_extractor, mock_llm):
    """Test successful mindset update."""
    # Create mock game state
    mock_game_state = {
        "game_id": "test-game-1",
        "players": ["player1", "player2"],
        "current_round": 1,
        "game_phase": "speaking",
        "phase_id": "1:speaking:abc123",
        "completed_speeches": [],
        "eliminated_players": [],
        "current_votes": {},
        "winner": None,
        "host_private_state": {
            "player_roles": {"player1": "spy", "player2": "civilian"},
            "civilian_word": "apple",
            "spy_word": "banana",
        },
        "player_private_states": {
            "player1": PlayerPrivateState(
                assigned_word="banana",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.5),
                    suspicions={},
                ),
            ),
        },
    }

    # Mock LLM response
    new_mindset = PlayerMindset(
        self_belief=SelfBelief(role="spy", confidence=0.8),
        suspicions={
            "player2": Suspicion(
                role="civilian", confidence=0.75, reason="Consistent with group"
            )
        },
    )

    mock_extractor_instance = Mock()
    mock_extractor_instance.invoke.return_value = {"responses": [new_mindset]}
    mock_extractor.return_value = mock_extractor_instance

    # Call tool
    result = update_player_mindset_tool(mock_game_state, "player1")

    # Verify result
    assert result is not None
    assert "player_private_states" in result
    assert "player1" in result["player_private_states"]

    updated_state = result["player_private_states"]["player1"]
    assert updated_state.playerMindset.self_belief.role == "spy"
    assert updated_state.playerMindset.self_belief.confidence == 0.8


def test_analyze_speech_consistency_access_control(mock_runtime):
    """Test access control for speech analysis."""
    with pytest.raises(ValueError, match="Invalid player_id"):
        analyze_speech_consistency(mock_runtime, "invalid_player", "player2")

    with pytest.raises(ValueError, match="Invalid target_player_id"):
        analyze_speech_consistency(mock_runtime, "player1", "invalid_target")


def test_analyze_speech_consistency_success(mock_runtime):
    """Test successful speech consistency analysis."""
    result = analyze_speech_consistency(mock_runtime, "player1", "player2")

    assert result is not None
    assert result["target_player_id"] == "player2"
    assert result["speech_count"] == 1
    assert len(result["speeches"]) == 1
    assert result["speeches"][0] == "It's something you can eat"
