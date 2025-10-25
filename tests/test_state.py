"""
Test coverage for game state structures and utility functions
"""

import pytest

from src.game.state import (
    GameState,
    Speech,
    Vote,
    SelfBelief,
    Suspicion,
    PlayerMindset,
    PlayerPrivateState,
    HostPrivateState,
    alive_players,
    next_alive_player,
    votes_ready,
    generate_phase_id,
    get_next_speech_seq,
    merge_votes,
    create_speech_record,
)


class TestStateStructures:
    """Test state structure definitions"""

    def test_speech_structure(self):
        """Test Speech TypedDict structure"""
        speech: Speech = {
            "round": 1,
            "seq": 0,
            "player_id": "player1",
            "content": "Test speech content",
            "ts": 1234567890,
        }
        assert speech["round"] == 1
        assert speech["seq"] == 0
        assert speech["player_id"] == "player1"
        assert speech["content"] == "Test speech content"
        assert speech["ts"] == 1234567890

    def test_vote_structure(self):
        """Test Vote structure (no abstention allowed)"""
        vote = Vote(
            target="player2",
            ts=1234567890,
            phase_id="1:voting:abc123",
        )
        assert vote.target == "player2"
        assert vote.ts == 1234567890
        assert vote.phase_id == "1:voting:abc123"

    def test_self_belief_structure(self):
        """Test SelfBelief structure"""
        belief = SelfBelief(role="civilian", confidence=0.8)
        assert belief.role == "civilian"
        assert belief.confidence == 0.8

    def test_player_private_state_structure(self):
        """Test PlayerPrivateState structure"""
        player_state = PlayerPrivateState(
            assigned_word="apple",
            playerMindset=PlayerMindset(
                self_belief=SelfBelief(role="civilian", confidence=0.8),
                suspicions={
                    "player2": Suspicion(role="spy", confidence=0.6, reason="test"),
                    "player3": Suspicion(
                        role="civilian", confidence=0.7, reason="test"
                    ),
                },
            ),
        )
        assert player_state.assigned_word == "apple"
        assert player_state.playerMindset.self_belief.role == "civilian"
        assert player_state.playerMindset.self_belief.confidence == 0.8
        assert player_state.playerMindset.suspicions["player2"].role == "spy"
        assert player_state.playerMindset.suspicions["player3"].role == "civilian"

    def test_host_private_state_structure(self):
        """Test HostPrivateState structure"""
        host_state: HostPrivateState = {
            "player_roles": {
                "player1": "civilian",
                "player2": "spy",
                "player3": "civilian",
            },
            "civilian_word": "apple",
            "spy_word": "banana",
        }
        assert host_state["player_roles"]["player2"] == "spy"
        assert host_state["civilian_word"] == "apple"
        assert host_state["spy_word"] == "banana"


class TestUtilityFunctions:
    """Test state utility functions"""

    @pytest.fixture
    def sample_state(self) -> GameState:
        """Create a sample game state for testing"""
        return {
            "game_id": "test_game",
            "players": ["player1", "player2", "player3", "player4"],
            "current_round": 1,
            "game_phase": "speaking",
            "phase_id": "1:speaking:abc123",
            "completed_speeches": [],
            "eliminated_players": [],
            "current_votes": {},
            "winner": None,
            "host_private_state": {
                "player_roles": {
                    "player1": "civilian",
                    "player2": "spy",
                    "player3": "civilian",
                    "player4": "civilian",
                },
                "civilian_word": "apple",
                "spy_word": "banana",
            },
            "player_private_states": {
                "player1": PlayerPrivateState(
                    assigned_word="apple",
                    playerMindset=PlayerMindset(
                        self_belief=SelfBelief(role="civilian", confidence=0.8),
                        suspicions={},
                    ),
                ),
                "player2": PlayerPrivateState(
                    assigned_word="banana",
                    playerMindset=PlayerMindset(
                        self_belief=SelfBelief(role="spy", confidence=0.9),
                        suspicions={},
                    ),
                ),
                "player3": PlayerPrivateState(
                    assigned_word="apple",
                    playerMindset=PlayerMindset(
                        self_belief=SelfBelief(role="civilian", confidence=0.7),
                        suspicions={},
                    ),
                ),
                "player4": PlayerPrivateState(
                    assigned_word="apple",
                    playerMindset=PlayerMindset(
                        self_belief=SelfBelief(role="civilian", confidence=0.6),
                        suspicions={},
                    ),
                ),
            },
        }

    def test_alive_players_all_alive(self, sample_state):
        """Test alive_players when no one is eliminated"""
        alive = alive_players(sample_state)
        assert set(alive) == {"player1", "player2", "player3", "player4"}
        assert len(alive) == 4

    def test_alive_players_with_eliminated(self, sample_state):
        """Test alive_players when some players are eliminated"""
        sample_state["eliminated_players"] = ["player2", "player4"]
        alive = alive_players(sample_state)
        assert set(alive) == {"player1", "player3"}
        assert len(alive) == 2

    def test_next_alive_player_round_start(self, sample_state):
        """Test next_alive_player at round start"""
        next_player = next_alive_player(sample_state)
        assert next_player == "player1"  # First player in order

    def test_next_alive_player_after_some_speeches(self, sample_state):
        """Test next_alive_player after some speeches"""
        sample_state["completed_speeches"] = [
            {
                "round": 1,
                "seq": 0,
                "player_id": "player1",
                "content": "Speech 1",
                "ts": 1234567890,
            },
            {
                "round": 1,
                "seq": 1,
                "player_id": "player2",
                "content": "Speech 2",
                "ts": 1234567891,
            },
        ]
        next_player = next_alive_player(sample_state)
        assert next_player == "player3"  # Next in order after player1 and player2

    def test_next_alive_player_all_spoken(self, sample_state):
        """Test next_alive_player when all players have spoken"""
        sample_state["completed_speeches"] = [
            {
                "round": 1,
                "seq": 0,
                "player_id": "player1",
                "content": "Speech 1",
                "ts": 1234567890,
            },
            {
                "round": 1,
                "seq": 1,
                "player_id": "player2",
                "content": "Speech 2",
                "ts": 1234567891,
            },
            {
                "round": 1,
                "seq": 2,
                "player_id": "player3",
                "content": "Speech 3",
                "ts": 1234567892,
            },
            {
                "round": 1,
                "seq": 3,
                "player_id": "player4",
                "content": "Speech 4",
                "ts": 1234567893,
            },
        ]
        next_player = next_alive_player(sample_state)
        assert next_player is None  # All players have spoken

    def test_votes_ready_no_votes(self, sample_state):
        """Test votes_ready when no votes exist"""
        sample_state["game_phase"] = "voting"
        sample_state["phase_id"] = "1:voting:xyz789"
        assert not votes_ready(sample_state)

    def test_votes_ready_all_voted_same_phase(self, sample_state):
        """Test votes_ready when all players voted in same phase"""
        sample_state["game_phase"] = "voting"
        sample_state["phase_id"] = "1:voting:xyz789"
        sample_state["current_votes"] = {
            "player1": Vote(
                target="player2",
                ts=1234567890,
                phase_id="1:voting:xyz789",
            ),
            "player2": Vote(
                target="player1",
                ts=1234567891,
                phase_id="1:voting:xyz789",
            ),
            "player3": Vote(
                target="player2",
                ts=1234567892,
                phase_id="1:voting:xyz789",
            ),
            "player4": Vote(
                target="player3",
                ts=1234567893,
                phase_id="1:voting:xyz789",
            ),
        }
        assert votes_ready(sample_state)

    def test_votes_ready_mixed_phase_ids(self, sample_state):
        """Test votes_ready with votes from different phases"""
        sample_state["game_phase"] = "voting"
        sample_state["phase_id"] = "1:voting:xyz789"
        sample_state["current_votes"] = {
            "player1": Vote(
                target="player2",
                ts=1234567890,
                phase_id="1:voting:xyz789",
            ),
            "player2": Vote(
                target="player1",
                ts=1234567891,
                phase_id="1:voting:xyz789",
            ),
            "player3": Vote(
                target="player2",
                ts=1234567892,
                phase_id="wrong_phase",
            ),  # Wrong phase
            "player4": Vote(
                target="player3",
                ts=1234567893,
                phase_id="1:voting:xyz789",
            ),
        }
        assert not votes_ready(sample_state)  # player3's vote is ignored

    def test_generate_phase_id(self, sample_state):
        """Test phase_id generation"""
        phase_id = generate_phase_id(sample_state)
        assert phase_id.startswith("1:speaking:")
        assert len(phase_id) > len("1:speaking:") + 4  # Should have some nonce

    def test_get_next_speech_seq_empty(self, sample_state):
        """Test get_next_speech_seq with no speeches"""
        next_seq = get_next_speech_seq(sample_state)
        assert next_seq == 0

    def test_get_next_speech_seq_with_speeches(self, sample_state):
        """Test get_next_speech_seq with existing speeches"""
        sample_state["completed_speeches"] = [
            {
                "round": 1,
                "seq": 0,
                "player_id": "player1",
                "content": "Speech 1",
                "ts": 1234567890,
            },
            {
                "round": 1,
                "seq": 1,
                "player_id": "player2",
                "content": "Speech 2",
                "ts": 1234567891,
            },
            {
                "round": 1,
                "seq": 2,
                "player_id": "player3",
                "content": "Speech 3",
                "ts": 1234567892,
            },
        ]
        next_seq = get_next_speech_seq(sample_state)
        assert next_seq == 3

    def test_get_next_speech_seq_different_rounds(self, sample_state):
        """Test get_next_speech_seq with speeches from different rounds"""
        sample_state["completed_speeches"] = [
            {
                "round": 1,
                "seq": 0,
                "player_id": "player1",
                "content": "Speech 1",
                "ts": 1234567890,
            },
            {
                "round": 1,
                "seq": 1,
                "player_id": "player2",
                "content": "Speech 2",
                "ts": 1234567891,
            },
            {
                "round": 2,
                "seq": 0,
                "player_id": "player1",
                "content": "Speech 3",
                "ts": 1234567892,
            },  # Different round
        ]
        next_seq = get_next_speech_seq(sample_state)
        assert next_seq == 2  # Only considers current round (1)


class TestStateConstraints:
    """Test state invariants and constraints"""

    def test_vote_target_not_none(self):
        """Test that Vote target cannot be None (no abstention)"""
        # This should raise a type error if we try to create a vote with None target
        vote = Vote(
            target="player2",  # Must be a string, not None
            ts=1234567890,
            phase_id="1:voting:abc123",
        )
        assert vote.target is not None
        assert isinstance(vote.target, str)

    def test_eliminated_players_subset(self):
        """Test that eliminated_players is always a subset of players"""
        state: GameState = {
            "game_id": "test",
            "players": ["p1", "p2", "p3"],
            "current_round": 1,
            "game_phase": "speaking",
            "phase_id": "1:speaking:abc",
            "completed_speeches": [],
            "eliminated_players": ["p2"],  # p2 is in players
            "current_votes": {},
            "winner": None,
            "host_private_state": {
                "player_roles": {},
                "civilian_word": "",
                "spy_word": "",
            },
            "player_private_states": {},
        }

        alive = alive_players(state)
        assert "p2" not in alive
        assert "p1" in alive
        assert "p3" in alive


class TestStateUpdateFunctions:
    """Test state update helper functions"""

    @pytest.fixture
    def sample_state(self) -> GameState:
        """Create a sample game state for testing"""
        return {
            "game_id": "test_game",
            "players": ["player1", "player2", "player3"],
            "current_round": 1,
            "game_phase": "speaking",
            "phase_id": "1:speaking:abc123",
            "completed_speeches": [],
            "eliminated_players": [],
            "current_votes": {},
            "winner": None,
            "host_private_state": {
                "player_roles": {
                    "player1": "civilian",
                    "player2": "spy",
                    "player3": "civilian",
                },
                "civilian_word": "apple",
                "spy_word": "banana",
            },
            "player_private_states": {
                "player1": PlayerPrivateState(
                    assigned_word="apple",
                    playerMindset=PlayerMindset(
                        self_belief=SelfBelief(role="civilian", confidence=0.8),
                        suspicions={},
                    ),
                ),
                "player2": PlayerPrivateState(
                    assigned_word="banana",
                    playerMindset=PlayerMindset(
                        self_belief=SelfBelief(role="spy", confidence=0.9),
                        suspicions={},
                    ),
                ),
                "player3": PlayerPrivateState(
                    assigned_word="apple",
                    playerMindset=PlayerMindset(
                        self_belief=SelfBelief(role="civilian", confidence=0.7),
                        suspicions={},
                    ),
                ),
            },
        }

    def test_merge_votes_same_phase(self):
        """Test merging votes from same phase"""
        current_votes = {
            "player1": Vote(target="player2", ts=1000, phase_id="1:voting:abc123")
        }
        new_votes = {
            "player2": Vote(target="player1", ts=1100, phase_id="1:voting:abc123"),
            "player3": Vote(target="player2", ts=1200, phase_id="1:voting:abc123"),
        }

        merged = merge_votes(current_votes, new_votes)

        assert len(merged) == 3
        assert merged["player1"].target == "player2"
        assert merged["player2"].target == "player1"
        assert merged["player3"].target == "player2"

    def test_merge_votes_different_phase(self):
        """Test merging votes from different phases (should be ignored)"""
        current_votes = {
            "player1": Vote(target="player2", ts=1000, phase_id="1:voting:abc123")
        }
        new_votes = {
            "player2": Vote(
                target="player1",
                ts=1100,
                phase_id="wrong_phase",
            ),  # Wrong phase
            "player3": Vote(target="player2", ts=1200, phase_id="1:voting:abc123"),
        }

        merged = merge_votes(current_votes, new_votes)

        assert len(merged) == 3  # All votes merged regardless of phase
        assert merged["player1"].target == "player2"
        assert merged["player2"].target == "player1"
        assert merged["player3"].target == "player2"

    def test_merge_votes_timestamp_conflict(self):
        """Test merging votes with timestamp conflict (choose latest)"""
        current_votes = {
            "player1": Vote(target="player2", ts=1000, phase_id="1:voting:abc123")
        }
        new_votes = {
            "player1": Vote(
                target="player3",
                ts=1500,
                phase_id="1:voting:abc123",
            )  # Later vote
        }

        merged = merge_votes(current_votes, new_votes)

        assert len(merged) == 1
        assert merged["player1"].target == "player3"  # Should use the later vote
        assert merged["player1"].ts == 1500

    def test_add_speech(self, sample_state):
        """Test adding speech with automatic seq and timestamp"""
        speech = create_speech_record(sample_state, "player1", "Test speech content")

        assert speech["round"] == 1
        assert speech["seq"] == 0  # First speech in round
        assert speech["player_id"] == "player1"
        assert speech["content"] == "Test speech content"
        assert isinstance(speech["ts"], int)
        assert speech["ts"] > 0

    def test_add_speech_sequential(self, sample_state):
        """Test adding multiple speeches with sequential seq"""
        # Add first speech
        speech1 = create_speech_record(sample_state, "player1", "First speech")
        assert speech1["seq"] == 0

        # Simulate adding the first speech to state
        sample_state["completed_speeches"].append(speech1)

        # Add second speech
        speech2 = create_speech_record(sample_state, "player2", "Second speech")
        assert speech2["seq"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
