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
    """Based on current phase and state, determine next node.
    - Speaking phase: Return next player's node name
    - Speaking ends: Enter voting phase, fan out to all players for voting
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
    """host_result conditional branch: continue or end"""
    return "end" if state.get("winner") else "continue"


def build_workflow_with_players(players: list[str], *, checkpointer=None):
    """Build workflow with specific players list."""
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

    # Build and run the workflow
    app = build_workflow_with_players(players)
    save_graph_image(app, filename="artifacts/agent_with_router.png")

    initial_state = {
        "game_id": f"game-{uuid4()}",
        "players": players,
        "game_phase": "setup",
    }

    langgraph_config = RunnableConfig(
        configurable={"thread_id": initial_state["game_id"]},
    )
    result = app.invoke(initial_state, config=langgraph_config)
    print(result)


if __name__ == "__main__":
    main()
