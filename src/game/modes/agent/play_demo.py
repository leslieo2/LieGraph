"""Command-line demo runner that showcases the agent behavior mode."""

from __future__ import annotations

import argparse
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from ...config import get_config
from ...graph import build_workflow_with_players
from ..registry import get_host_behavior, get_player_behavior


def run_demo(*, behavior_mode: str = "agent", players: list[str] | None = None) -> None:
    """Execute a single demo game and print agent strategy summaries."""

    config = get_config()
    player_names = players or config.generate_player_names()

    app = build_workflow_with_players(player_names)
    game_id = f"demo-{uuid4()}"

    initial_state = {
        "game_id": game_id,
        "players": player_names,
        "game_phase": "setup",
        "behavior_mode": behavior_mode,
    }
    runnable_config = RunnableConfig(configurable={"thread_id": game_id})

    print(
        f"🎮 Running demo game {game_id} in {behavior_mode} mode with players: {player_names}"
    )
    result_state = app.invoke(initial_state, config=runnable_config)
    print("🏁 Demo completed. Final state summary:")
    print(
        {
            "current_round": result_state.get("current_round"),
            "phase": result_state.get("game_phase"),
            "winner": result_state.get("winner"),
            "eliminated_players": result_state.get("eliminated_players"),
        }
    )

    _print_host_journal(behavior_mode)
    _print_player_strategy_log(behavior_mode, player_names)


def _print_host_journal(behavior_mode: str) -> None:
    host_behavior = get_host_behavior(mode=behavior_mode)
    journal = getattr(host_behavior, "journal", None)
    if not journal:
        print("🗒️  Host journal not available for this mode.")
        return

    print("🗒️  Host journal:")
    for entry in journal:
        print(f"  - [{entry.ts.isoformat()}] {entry.action}: {entry.payload}")


def _print_player_strategy_log(behavior_mode: str, players: list[str]) -> None:
    player_behavior = get_player_behavior(players[0], mode=behavior_mode)
    if not hasattr(player_behavior, "memory_for"):
        print("🧠 Player strategy logs unavailable for this mode.")
        return

    print("🧠 Player strategy snapshots:")
    for player_id in players:
        memory = player_behavior.memory_for(player_id)
        if not memory.decisions:
            continue
        latest_decision = memory.decisions[-1]
        print(
            f"  - Player {player_id}: "
            f"{latest_decision.kind} strategy='{latest_decision.strategy}' "
            f"target={latest_decision.target}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agent-mode demo game.")
    parser.add_argument(
        "--mode",
        default="agent",
        choices=["agent", "workflow"],
        help="Behavior mode to use for the demo.",
    )
    parser.add_argument(
        "--players",
        nargs="*",
        help="Optional explicit list of player IDs to use.",
    )
    args = parser.parse_args()
    run_demo(behavior_mode=args.mode, players=args.players)


if __name__ == "__main__":
    main()
