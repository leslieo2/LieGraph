"""
LangGraph workflow orchestrator for the "Who Is Spy" game.

This module defines the main state machine that orchestrates the game flow:
- Game setup and role assignment
- Sequential speaking phases
- Concurrent voting phases
- Game state transitions and win condition checking

The workflow is built using LangGraph's StateGraph with conditional routing
between different phases of the game.

Architecture:
- StateGraph manages the overall game state transitions
- Conditional edges route between speaking and voting phases
- Concurrent execution for voting, sequential for speaking
- Private state management for player mindsets and game setup
"""

from functools import partial
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import END, StateGraph

from src.game.nodes.host import host_setup, host_stage_switch, host_result
from src.game.nodes.player import player_speech, player_vote
from src.game.nodes.transition import check_votes_and_transition
from src.game.state import GameState, votes_ready, next_alive_player
from src.tools import save_graph_image
from src.game.config import get_config


def route_from_stage(state: GameState) -> list[str] | str:
    """Route to appropriate nodes based on current game phase.

    This is the main conditional routing function that determines the next
    step in the game flow based on the current phase.

    Args:
        state: Current game state containing phase information

    Returns:
        Single node name for speaking phase, or list of node names for voting phase

    Routing Logic:
        - Speaking phase: Routes to next player's speech node
        - Voting phase: Routes to all alive players' vote nodes concurrently
        - Unknown phase: Falls back to host_result for error handling
    """
    game_phase = state.get("game_phase")
    if game_phase == "speaking":
        pid = next_alive_player(state)
        return f"player_speech_{pid}"
    elif game_phase == "voting":
        # When entering voting phase, only alive players vote concurrently
        alive_players = [
            p for p in state["players"] if p not in state.get("eliminated_players", [])
        ]
        return [f"player_vote_{pid}" for pid in alive_players]
    else:
        # Fault tolerance: Unknown phase returns to host_result for decision
        return "host_result"


def should_continue(state: GameState) -> str:
    """Determine if the game should continue or end.

    This conditional function is used by the host_result node to decide
    whether the game should proceed to another round or end.

    Args:
        state: Current game state

    Returns:
        "end" if there's a winner, "continue" otherwise
    """
    return "end" if state.get("winner") else "continue"


def build_workflow_with_players(players: list[str], *, checkpointer=None):
    """Build the complete LangGraph workflow for a specific set of players.

    This function constructs the entire state machine with all nodes and edges
    configured for the given player list.

    Args:
        players: List of player IDs to include in the game
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled LangGraph application ready for execution

    Node Architecture:
        - Host nodes: setup, stage switching, result handling
        - Player nodes: speech and vote nodes for each player
        - Transition nodes: vote counting and phase transitions
    """
    workflow = StateGraph(GameState)

    # Register nodes
    workflow.add_node("host_setup", host_setup)
    workflow.add_node(
        "host_stage_switch", host_stage_switch
    )  # Responsible for writing phase/next_* pointers
    workflow.add_node("host_result", host_result)

    workflow.add_node("check_votes_and_transition", check_votes_and_transition)

    for pid in players:
        workflow.add_node(f"player_speech_{pid}", partial(player_speech, player_id=pid))
        workflow.add_node(f"player_vote_{pid}", partial(player_vote, player_id=pid))

    # Basic skeleton
    workflow.add_edge(START, "host_setup")
    workflow.add_edge("host_setup", "host_stage_switch")

    # Conditional routing from host_stage_switch to: next speaker or voting phase
    workflow.add_conditional_edges(
        "host_stage_switch",
        route_from_stage,
        {
            **{f"player_speech_{pid}": f"player_speech_{pid}" for pid in players},
            **{f"player_vote_{pid}": f"player_vote_{pid}" for pid in players},
            "host_result": "host_result",  # Fallback for unknown phase
        },
    )

    # Each speech node returns to host_stage_switch (advance to next speaker or switch to voting)
    for pid in players:
        workflow.add_edge(f"player_speech_{pid}", "host_stage_switch")

    # Voting phase: All player votes converge to check_votes_and_transition
    for pid in players:
        workflow.add_edge(f"player_vote_{pid}", "check_votes_and_transition")

    # check_votes_and_transition conditional edges: if all votes are ready, enter host_result, otherwise wait
    workflow.add_conditional_edges(
        "check_votes_and_transition",
        lambda state: "host_result" if votes_ready(state) else "__continue__",
        {"host_result": "host_result"},
    )

    # host_result conditional edges: continue game or end
    workflow.add_conditional_edges(
        "host_result",
        should_continue,
        {"continue": "host_stage_switch", "end": END},
    )

    memory = checkpointer or MemorySaver()
    app = workflow.compile(checkpointer=memory)
    # Set default recursion limit for the compiled app
    app = app.with_config({"recursion_limit": 500})
    return app


def build_workflow(config=None):
    """Build workflow for LangGraph Server - accepts RunnableConfig parameter.

    For LangGraph Server, we build a workflow using the player count from config.yaml.
    The frontend will get the actual player list from the game state.
    """
    # Load configuration to get the configured player count
    game_config = get_config()

    # Generate player names based on configuration
    players = game_config.generate_player_names()

    print(f"🎮 Building workflow with {len(players)} players: {players}")

    return build_workflow_with_players(players)


def main():
    """Main execution function using configuration."""
    # Load configuration
    config = get_config()

    # Generate player names based on configuration
    players = config.generate_player_names()

    print(f"Game Configuration:")
    print(f"  Player count: {config.player_count}")
    print(f"  Players: {players}")
    print(f"  Vocabulary pairs: {len(config.vocabulary)}")
    print(f"  Behavior mode: {config.behavior_mode}")

    # Build and run the workflow
    app = build_workflow_with_players(players)
    save_graph_image(app, filename="artifacts/agent_with_router.png")

    initial_state = {
        "game_id": f"game-{uuid4()}",
        "players": players,
        "game_phase": "setup",
        "behavior_mode": config.behavior_mode,
    }

    langgraph_config = RunnableConfig(
        configurable={"thread_id": initial_state["game_id"]},
    )
    result = app.invoke(initial_state, config=langgraph_config)
    print(result)


if __name__ == "__main__":
    main()
