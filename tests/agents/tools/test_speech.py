"""
Unit tests for speech generation tools.

Tests cover:
- generate_speech_tool functionality
- Access control validation
- Speech sanitization
- Clarity determination
"""

import pytest
from unittest.mock import Mock, patch
from src.game.agents.tools.speech import (
    generate_speech_tool,
    analyze_speech_consistency,
    _determine_clarity,
    _sanitize_speech_output,
)
from src.game.state import (
    PlayerPrivateState,
    PlayerMindset,
    SelfBelief,
)


@pytest.fixture
def mock_game_state():
    """Create a mock game state for testing."""
    return {
        "game_id": "test-game-1",
        "players": ["player1", "player2", "player3"],
        "current_round": 2,
        "game_phase": "speaking",
        "phase_id": "2:speaking:xyz789",
        "completed_speeches": [],
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
                    self_belief=SelfBelief(role="spy", confidence=0.8),
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
        },
    }


@pytest.fixture
def mock_runtime(mock_game_state):
    """Create a mock ToolRuntime."""
    # Not needed anymore since we pass state directly
    return mock_game_state


def test_generate_speech_tool_access_control(mock_runtime):
    """Test that invalid player_id raises ValueError."""
    with pytest.raises(ValueError, match="Invalid player_id"):
        generate_speech_tool(mock_runtime, "invalid_player")


def test_generate_speech_tool_wrong_phase(mock_runtime):
    """Test that tool returns empty update when not in speaking phase."""
    mock_runtime["game_phase"] = "voting"

    result = generate_speech_tool(mock_runtime, "player1")

    assert result is not None
    assert result == {}


@patch("src.tools.llm.create_llm")
def test_generate_speech_tool_success(mock_llm):
    """Test successful speech generation."""
    # Create mock game state
    mock_game_state = {
        "game_id": "test-game-1",
        "players": ["player1", "player2"],
        "current_round": 2,
        "game_phase": "speaking",
        "phase_id": "2:speaking:xyz789",
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
                    self_belief=SelfBelief(role="spy", confidence=0.8),
                    suspicions={},
                ),
            ),
        },
    }

    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "It's yellow and sweet"
    mock_llm_instance = Mock()
    mock_llm_instance.invoke.return_value = mock_response
    mock_llm.return_value = mock_llm_instance

    result = generate_speech_tool(mock_game_state, "player1")

    assert result is not None
    assert "completed_speeches" in result
    assert len(result["completed_speeches"]) == 1

    speech = result["completed_speeches"][0]
    assert speech["player_id"] == "player1"
    assert speech["content"] == "It's yellow and sweet"
    assert speech["round"] == 2


def test_determine_clarity_spy():
    """Test clarity determination for spy role."""
    # Early rounds - low clarity
    clarity, desc = _determine_clarity("spy", 0.8, 1)
    assert clarity == "low"
    assert "LOW clarity" in desc

    # Mid rounds - medium clarity
    clarity, desc = _determine_clarity("spy", 0.8, 3)
    assert clarity == "medium"
    assert "MEDIUM clarity" in desc

    # Late rounds - still medium
    clarity, desc = _determine_clarity("spy", 0.8, 5)
    assert clarity == "medium"
    assert "MEDIUM clarity" in desc


def test_determine_clarity_civilian():
    """Test clarity determination for civilian role."""
    # Round 1 - low clarity
    clarity, desc = _determine_clarity("civilian", 0.7, 1)
    assert clarity == "low"
    assert "LOW clarity" in desc

    # Round 2 - medium clarity
    clarity, desc = _determine_clarity("civilian", 0.7, 2)
    assert clarity == "medium"
    assert "MEDIUM clarity" in desc

    # Round 3+ - high clarity
    clarity, desc = _determine_clarity("civilian", 0.7, 3)
    assert clarity == "high"
    assert "HIGH clarity" in desc


def test_sanitize_speech_output():
    """Test speech output sanitization."""
    # Test emoji removal
    assert _sanitize_speech_output("Hello 😊 world") == "Hello world"

    # Test whitespace normalization
    assert _sanitize_speech_output("Hello   world\n\n") == "Hello world"

    # Test multi-line to single line
    assert _sanitize_speech_output("Line 1\nLine 2\nLine 3") == "Line 3"

    # Test None input
    assert _sanitize_speech_output(None) == ""


def test_analyze_speech_consistency_tool(mock_runtime):
    """Test speech consistency analysis."""
    # Add some speeches
    mock_runtime["completed_speeches"] = [
        {
            "round": 1,
            "seq": 0,
            "player_id": "player2",
            "content": "It's red and sweet",
            "ts": 1234567890,
        },
        {
            "round": 1,
            "seq": 1,
            "player_id": "player2",
            "content": "It's a fruit",
            "ts": 1234567891,
        },
    ]

    result = analyze_speech_consistency(mock_runtime, "player1", "player2")

    assert result is not None
    assert result["target_player_id"] == "player2"
    assert result["speech_count"] == 2
    assert len(result["target_speeches"]) == 2
