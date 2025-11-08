from typing import Dict, Any, cast

from ..config import get_config
from ..metrics import metrics_collector
from ..state import GameState, next_alive_player, generate_phase_id
from ..rules import (
    assign_roles_and_words,
    calculate_eliminated_player,
    determine_winner,
)
from ..logger import get_logger
from .helpers import get_assigned_word

logger = get_logger(__name__)


def host_setup(state: GameState) -> Dict[str, Any]:
    """Initializes the game, assigning roles and words."""
    config = get_config()
    desired_state = config.metrics_enabled or metrics_collector.enabled
    metrics_collector.set_enabled(desired_state)

    player_list = state.get("players")

    if not player_list:
        player_list = config.generate_player_names()

    # Pass the existing host_private_state to assign_roles_and_words
    # This allows custom words from frontend to be used if provided
    host_private_state = state.get("host_private_state", {})
    assignments = assign_roles_and_words(
        player_list, host_private_state=host_private_state
    )

    logger.info("Host initializing game with %d players", len(player_list))
    logger.debug("Initial player list: %s", player_list)
    for player_id, private_state in assignments["player_private_states"].items():
        assigned_word = get_assigned_word(private_state)
        logger.debug("Assigned word for %s: %s", player_id, assigned_word)

    if metrics_collector.enabled:
        metrics_collector.on_game_start(
            game_id=state.get("game_id"),
            players=player_list,
            player_roles=assignments["host_private_state"]["player_roles"],
        )

    # These private states will be merged by the graph runner
    game_setup_state = {
        "current_round": 1,
        "game_phase": "speaking",
        "host_private_state": assignments["host_private_state"],
        "player_private_states": assignments["player_private_states"],
        "current_votes": {},
        "completed_speeches": [],
        "eliminated_players": [],
        "winner": None,
        "players": player_list,
    }
    game_setup_state["phase_id"] = generate_phase_id(game_setup_state)
    return game_setup_state


def host_stage_switch(state: GameState) -> Dict[str, Any]:
    """
    Determines the next speaker or transitions to the voting phase if speaking is complete.
    The graph's routing logic will use the new phase to decide where to go next.
    """
    if state["game_phase"] == "speaking":
        next_player = next_alive_player(state)
        if next_player:
            # There's another player to speak
            logger.info("Stage switch selecting next speaker: %s", next_player)
            return (
                {}
            )  # No state update needed - next_alive_player is computed dynamically
        else:
            # All players have spoken, transition to voting
            logger.info("Stage switch detected all speeches complete; starting voting")
            updates = {"game_phase": "voting", "current_votes": {}}
            updates["phase_id"] = generate_phase_id(
                state | updates
            )  # Generate new phase_id for voting
            return updates
    # No state change needed otherwise, the graph will continue routing.
    return {}


def host_result(state: GameState) -> Dict[str, Any]:
    """
    Calculates the result of a round, eliminates a player, and checks for a winner.
    This node is the aggregation point after voting.
    """
    eliminated_player = calculate_eliminated_player(state)

    logger.info(
        "Host round %d voted out player: %s",
        state["current_round"],
        eliminated_player,
    )
    logger.debug("Current votes: %s", state.get("current_votes", {}))

    # Create a temporary state to check for a winner after the potential elimination
    temp_state = cast(GameState, state.copy())
    if eliminated_player:
        # Use the reducer 'add' format by providing a list with the new item
        temp_state["eliminated_players"] = state.get("eliminated_players", []) + [
            eliminated_player
        ]

    winner = determine_winner(temp_state, state["host_private_state"])

    if winner:
        logger.info("Winner determined: %s", winner)
        if metrics_collector.enabled:
            metrics_collector.on_game_end(
                game_id=state.get("game_id"),
                winner=winner,
            )
        update = {"game_phase": "result", "winner": winner}
        if eliminated_player:
            update["eliminated_players"] = [eliminated_player]
        return update

    # No winner, advance to the next round
    logger.info("No winner; advancing to round %d", state["current_round"] + 1)
    return _prepare_next_round(state, eliminated_player)


def _prepare_next_round(state: GameState, eliminated: str | None) -> Dict[str, Any]:
    """Prepares the state for the next round."""
    updates = {
        "game_phase": "speaking",
        "current_round": state["current_round"] + 1,
        "current_votes": {},  # Clear votes for the new round
    }
    updates["phase_id"] = generate_phase_id(
        state | updates
    )  # Generate new phase_id for speaking
    if eliminated:
        updates["eliminated_players"] = [eliminated]

    logger.info("Advancing to round %d", updates["current_round"])
    if eliminated:
        logger.info("Eliminated player carried forward: %s", eliminated)
    return updates
