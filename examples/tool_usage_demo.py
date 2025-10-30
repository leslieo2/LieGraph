"""
Example script demonstrating the usage of agent tools.

This script shows how to:
1. Set up a mock game state
2. Call individual tools for testing
3. Demonstrate tool capabilities and access control

Usage:
    python examples/tool_usage_demo.py
"""

from unittest.mock import Mock
from src.game.agents.tools.identity import (
    update_player_mindset_tool,
    analyze_speech_consistency,
)
from src.game.agents.tools.speech import generate_speech_tool
from src.game.agents.tools.voting import decide_vote_tool, analyze_voting_patterns
from src.game.state import (
    PlayerPrivateState,
    PlayerMindset,
    SelfBelief,
    Suspicion,
    Vote,
)


def create_demo_game_state():
    """Create a demo game state for testing."""
    return {
        "game_id": "demo-game-1",
        "players": ["alice", "bob", "charlie"],
        "current_round": 2,
        "game_phase": "speaking",
        "phase_id": "2:speaking:demo123",
        "completed_speeches": [
            {
                "round": 1,
                "seq": 0,
                "player_id": "alice",
                "content": "It's something sweet and fruity",
                "ts": 1234567890000,
            },
            {
                "round": 1,
                "seq": 1,
                "player_id": "bob",
                "content": "It's red and crunchy",
                "ts": 1234567891000,
            },
            {
                "round": 1,
                "seq": 2,
                "player_id": "charlie",
                "content": "It grows on trees",
                "ts": 1234567892000,
            },
        ],
        "eliminated_players": [],
        "current_votes": {},
        "winner": None,
        "host_private_state": {
            "player_roles": {"alice": "spy", "bob": "civilian", "charlie": "civilian"},
            "civilian_word": "apple",
            "spy_word": "banana",
        },
        "player_private_states": {
            "alice": PlayerPrivateState(
                assigned_word="banana",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.6),
                    suspicions={},
                ),
            ),
            "bob": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.75),
                    suspicions={
                        "alice": Suspicion(
                            role="spy",
                            confidence=0.65,
                            reason="Her description seems slightly off",
                        ),
                    },
                ),
            ),
            "charlie": PlayerPrivateState(
                assigned_word="apple",
                playerMindset=PlayerMindset(
                    self_belief=SelfBelief(role="civilian", confidence=0.7),
                    suspicions={},
                ),
            ),
        },
    }


def demo_speech_consistency_analysis():
    """Demo: Analyze speech consistency."""
    print("\n" + "=" * 60)
    print("DEMO: Speech Consistency Analysis")
    print("=" * 60)

    state = create_demo_game_state()

    # Bob analyzes Alice's speeches
    print("\n[Bob analyzing Alice's speeches...]")
    result = analyze_speech_consistency(state, "bob", "alice")

    print(f"\nAnalysis Results:")
    print(f"  Target: {result['target_player_id']}")
    print(f"  Speech Count: {result['speech_count']}")
    print(f"  Speeches: {result['target_speeches']}")
    print(f"  Average Length: {result['avg_speech_length']:.1f} characters")


def demo_voting_patterns():
    """Demo: Analyze voting patterns."""
    print("\n" + "=" * 60)
    print("DEMO: Voting Pattern Analysis")
    print("=" * 60)

    state = create_demo_game_state()
    state["game_phase"] = "voting"
    state["phase_id"] = "2:voting:demo456"

    # Add some votes
    state["current_votes"] = {
        "alice": Vote(target="bob", ts=1234567900000, phase_id="2:voting:demo456"),
        "bob": Vote(target="alice", ts=1234567901000, phase_id="2:voting:demo456"),
        "charlie": Vote(target="alice", ts=1234567902000, phase_id="2:voting:demo456"),
    }

    print("\n[Analyzing voting patterns...]")
    result = analyze_voting_patterns(state, "bob")

    print(f"\nVoting Pattern Results:")
    print(f"  Total Votes: {result['total_votes']}")
    print(f"  Vote Distribution: {result['vote_distribution']}")
    print(f"  Most Voted Player: {result['most_voted_player']}")
    print(f"  Most Voted Count: {result['most_voted_count']}")
    print(f"  Is Bandwagon: {result['is_bandwagon']}")


def demo_access_control():
    """Demo: Access control validation."""
    print("\n" + "=" * 60)
    print("DEMO: Access Control Validation")
    print("=" * 60)

    state = create_demo_game_state()

    # Try to analyze with invalid player ID
    print("\n[Attempting to use invalid player ID...]")
    try:
        analyze_speech_consistency(state, "invalid_player", "alice")
        print("  ❌ Access control FAILED - no error raised!")
    except ValueError as e:
        print(f"  ✅ Access control PASSED - Error: {e}")

    # Try to analyze invalid target
    print("\n[Attempting to analyze invalid target...]")
    try:
        analyze_speech_consistency(state, "bob", "invalid_target")
        print("  ❌ Access control FAILED - no error raised!")
    except ValueError as e:
        print(f"  ✅ Access control PASSED - Error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AGENT TOOLS USAGE DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows the capabilities of the new agent tool library.")
    print("Phase 1 implementation includes:")
    print("  - Identity inference tools")
    print("  - Speech analysis tools")
    print("  - Voting decision tools")
    print("  - Proper access control")

    try:
        demo_speech_consistency_analysis()
        demo_voting_patterns()
        demo_access_control()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nNext Steps:")
        print("  1. Run unit tests: pytest tests/agents/tools/")
        print("  2. Integrate tools into agent factories (Phase 2)")
        print("  3. Update graph nodes to use agents (Phase 3)")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
