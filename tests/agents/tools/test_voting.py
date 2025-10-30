"""
Unit tests for voting decision tools.

Tests cover:
- decide_vote_tool functionality
- Vote decision logic
- Access control validation
- Voting pattern analysis
"""

import pytest
from unittest.mock import Mock
from src.game.agents.tools.voting import (
    decide_vote_tool,
    analyze_voting_patterns,
)
from src.game.state import (
    PlayerPrivateState,
    PlayerMindset,
    SelfBelief,
    Suspicion,
    Vote,
)


@pytest.fixture
def mock_game_state():
    """Create a mock game state for testing."""
    return {
        "game_id": "test-game-1",
        "players": ["player1", "player2", "player3"],
        "current_round": 2,
        "game_phase": "voting",
        "phase_id": "2:voting:xyz789",
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
                    suspicions={
                        "player2": Suspicion(
                            role="civilian", confidence=0.75, reason="Consistent"
                        ),
                        "player3": Suspicion(
                            role="spy", confidence=0.6, reason="Vague speech"
                        ),
                    },
                ),
            ),
            "player2": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.7),
                    suspicions={
                        "player1": Suspicion(
                            role="spy", confidence=0.8, reason="Outlier"
                        ),
                    },
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


def test_decide_vote_tool_access_control(mock_runtime):
    """Test that invalid player_id raises ValueError."""
    with pytest.raises(ValueError, match="Invalid player_id"):
        decide_vote_tool(mock_runtime, "invalid_player")


def test_decide_vote_tool_wrong_phase(mock_runtime):
    """Test that tool returns empty update when not in voting phase."""
    mock_runtime["game_phase"] = "speaking"

    result = decide_vote_tool(mock_runtime, "player1")

    assert result is not None
    assert result == {}


def test_decide_vote_tool_success(mock_runtime):
    """Test successful vote decision."""
    result = decide_vote_tool(mock_runtime, "player1")

    assert result is not None
    assert "current_votes" in result
    assert "player1" in result["current_votes"]

    vote = result["current_votes"]["player1"]
    assert isinstance(vote, Vote)
    # Player1 is spy with high confidence, so should vote for someone they suspect is spy
    # or someone they don't trust (different role alignment)
    assert vote.target in ["player2", "player3"]
    assert vote.phase_id == "2:voting:xyz789"


def test_decide_vote_tool_low_confidence(mock_runtime):
    """Test vote decision with low confidence (role reversal)."""
    # Set player1's confidence below 0.5
    mock_runtime["player_private_states"][
        "player1"
    ].playerMindset.self_belief.confidence = 0.4

    result = decide_vote_tool(mock_runtime, "player1")

    assert result is not None
    assert "current_votes" in result

    vote = result["current_votes"]["player1"]
    assert vote.target in ["player2", "player3"]


def test_decide_vote_tool_no_suspicions(mock_runtime):
    """Test vote decision when player has no suspicions."""
    # Clear suspicions for player3
    result = decide_vote_tool(mock_runtime, "player3")

    assert result is not None
    assert "current_votes" in result

    vote = result["current_votes"]["player3"]
    # Should vote for first available other player
    assert vote.target in ["player1", "player2"]


def test_decide_vote_tool_eliminated_player(mock_runtime):
    """Test that eliminated player cannot vote."""
    mock_runtime["eliminated_players"] = ["player1"]

    result = decide_vote_tool(mock_runtime, "player1")

    assert result is not None
    assert result == {}


def test_analyze_voting_patterns(mock_runtime):
    """Test voting pattern analysis."""
    # Add some votes
    mock_runtime["current_votes"] = {
        "player1": Vote(target="player2", ts=1234567890, phase_id="2:voting:xyz789"),
        "player2": Vote(target="player1", ts=1234567891, phase_id="2:voting:xyz789"),
        "player3": Vote(target="player2", ts=1234567892, phase_id="2:voting:xyz789"),
    }

    result = analyze_voting_patterns(mock_runtime, "player1")

    assert result is not None
    assert result["total_votes"] == 3
    assert result["vote_distribution"]["player2"] == 2
    assert result["vote_distribution"]["player1"] == 1
    assert result["most_voted_player"] == "player2"
    assert result["most_voted_count"] == 2
    assert result["is_bandwagon"] is True  # 2 > 3/2


def test_analyze_voting_patterns_no_votes(mock_runtime):
    """Test voting pattern analysis with no votes."""
    result = analyze_voting_patterns(mock_runtime, "player1")

    assert result is not None
    assert result["total_votes"] == 0
    assert result["vote_distribution"] == {}
    assert result["most_voted_player"] is None
    assert result["most_voted_count"] == 0
    assert result["is_bandwagon"] is False
