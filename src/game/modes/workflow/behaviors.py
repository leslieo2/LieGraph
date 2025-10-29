"""Workflow-aligned behavior implementations for host and player nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, cast

from ...config import get_config
from src.tools.llm import get_default_llm_client
from src.tools.llm.inference import llm_update_player_mindset
from src.tools.llm.speech import llm_generate_speech
from ...metrics import metrics_collector
from ...rules import (
    assign_roles_and_words,
    calculate_eliminated_player,
    determine_winner,
)
from ...state import (
    GameState,
    PlayerMindset,
    PlayerPrivateState,
    Speech,
    Vote,
    alive_players,
    create_speech_record,
    generate_phase_id,
    get_player_context,
    merge_probs,
    next_alive_player,
)
from ..shared.interfaces import (
    BehaviorResult,
    HostBehavior,
    HostNodeContext,
    PlayerBehavior,
    PlayerNodeContext,
)


@dataclass(slots=True)
class WorkflowHostBehavior(HostBehavior):
    """Workflow behavior that mirrors the existing host node logic."""

    def setup(self, ctx: HostNodeContext) -> BehaviorResult:
        state = ctx.state
        player_list = state.get("players")

        if not player_list:
            config = get_config()
            player_list = config.generate_player_names()

        host_private_state = state.get("host_private_state", {})
        assignments = assign_roles_and_words(
            player_list, host_private_state=host_private_state
        )

        print(f"🎮 Host: Initializing game, {len(player_list)} players")
        print(f"   Players: {player_list}")
        for player_id, private_state in assignments["player_private_states"].items():
            print(
                f"   Player {player_id}: Assigned word = {private_state.assigned_word}"
            )

        metrics_collector.on_game_start(
            game_id=state.get("game_id"),
            players=player_list,
            player_roles=assignments["host_private_state"]["player_roles"],
        )

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

    def stage_switch(self, ctx: HostNodeContext) -> BehaviorResult:
        state = ctx.state
        if state["game_phase"] == "speaking":
            next_player = next_alive_player(state)
            if next_player:
                print(f"🎮 Stage switch: Next speaker is {next_player}")
                return {}
            print("🎮 Stage switch: All players have spoken, starting voting")
            updates = {"game_phase": "voting", "current_votes": {}}
            updates["phase_id"] = generate_phase_id(state | updates)
            return updates
        return {}

    def resolve_round(self, ctx: HostNodeContext) -> BehaviorResult:
        state = ctx.state
        eliminated_player = calculate_eliminated_player(state)

        print(
            f"🎮 Host Round {state['current_round']} - Voted out player: {eliminated_player}"
        )
        print(f"   Current votes: {state.get('current_votes', {})}")

        temp_state = cast(GameState, state.copy())
        if eliminated_player:
            temp_state["eliminated_players"] = state.get("eliminated_players", []) + [
                eliminated_player
            ]

        winner = determine_winner(temp_state, state["host_private_state"])

        if winner:
            print(f"🎮 Host announces result: Game over! Winner: {winner}")
            metrics_collector.on_game_end(
                game_id=state.get("game_id"),
                winner=winner,
            )
            update: BehaviorResult = {
                "game_phase": "result",
                "winner": winner,
            }
            if eliminated_player:
                update["eliminated_players"] = [eliminated_player]
            return update

        print(f"🎮 Game not over: Round {state['current_round'] + 1}")
        return self._prepare_next_round(state, eliminated_player)

    @staticmethod
    def _prepare_next_round(
        state: GameState, eliminated: Optional[str]
    ) -> BehaviorResult:
        updates: BehaviorResult = {
            "game_phase": "speaking",
            "current_round": state["current_round"] + 1,
            "current_votes": {},
        }
        updates["phase_id"] = generate_phase_id(state | updates)
        if eliminated:
            updates["eliminated_players"] = [eliminated]

        print(f"🎮 ADVANCE ROUND: Moving to round {updates['current_round']}")
        if eliminated:
            print(f"   Voted out player: {eliminated}")
        return updates


class WorkflowPlayerBehavior(PlayerBehavior):
    """Workflow behavior that mirrors the existing player node logic."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client
        self._use_custom_client = llm_client is not None

    @property
    def llm_client(self) -> Any:
        """Get the LLM client, using custom client if provided, otherwise lazy-loading default."""
        if self._use_custom_client:
            return self._llm_client
        return get_default_llm_client()

    def decide_speech(self, ctx: PlayerNodeContext) -> BehaviorResult:
        state = ctx.state
        player_id = ctx.player_id

        if state["game_phase"] != "speaking":
            return {}
        if player_id not in alive_players(state):
            return {}

        player_context, my_word = self._get_player_context(state, player_id)

        print(
            f"🎤 PLAYER SPEECH: {player_id} is generating speech for round {state['current_round']}"
        )
        print(f"   Assigned word: {my_word}")

        config = get_config()
        private_state = player_context["private"]
        existing_player_mindset = private_state.playerMindset

        update_mindset_fn = ctx.extras.get(
            "llm_update_player_mindset", llm_update_player_mindset
        )
        generate_speech_fn = ctx.extras.get("llm_generate_speech", llm_generate_speech)

        updated_mindset = update_mindset_fn(
            llm_client=self.llm_client,
            my_word=my_word,
            completed_speeches=state["completed_speeches"],
            players=state["players"],
            alive=alive_players(state),
            me=player_id,
            rules=config.get_game_rules(),
            existing_player_mindset=existing_player_mindset,
        )

        metrics_collector.on_player_mindset_update(
            game_id=ctx.state.get("game_id"),
            round_number=ctx.state["current_round"],
            phase=ctx.state["game_phase"],
            player_id=ctx.player_id,
            mindset=updated_mindset,
        )

        speech_text = generate_speech_fn(
            llm_client=self.llm_client,
            my_word=my_word,
            self_belief=updated_mindset.self_belief,
            suspicions=updated_mindset.suspicions,
            completed_speeches=state["completed_speeches"],
            me=player_id,
            alive=alive_players(state),
            current_round=state["current_round"],
        )

        speech_lines = (speech_text or "").splitlines() or [""]
        print("   Speech content:")
        for line in speech_lines:
            print(f"      {line}")

        speech_record: Speech = create_speech_record(
            state=ctx.state,
            player_id=player_id,
            content=speech_text,
        )

        delta_private = self._create_player_private_state_delta(
            updated_mindset,
            player_context,
            my_word,
        )

        return {
            "completed_speeches": [speech_record],
            "player_private_states": {player_id: delta_private},
        }

    def decide_vote(self, ctx: PlayerNodeContext) -> BehaviorResult:
        state = ctx.state
        player_id = ctx.player_id

        if state["game_phase"] != "voting":
            return {}
        if player_id not in alive_players(state):
            return {}

        player_context, my_word = self._get_player_context(state, player_id)

        print(f"🗳️ PLAYER VOTE: {player_id} is voting in round {state['current_round']}")

        config = get_config()
        private_state = player_context["private"]
        existing_player_mindset = private_state.playerMindset

        update_mindset_fn = ctx.extras.get(
            "llm_update_player_mindset", llm_update_player_mindset
        )

        updated_mindset = update_mindset_fn(
            llm_client=self.llm_client,
            my_word=my_word,
            completed_speeches=state["completed_speeches"],
            players=state["players"],
            alive=alive_players(state),
            me=player_id,
            rules=config.get_game_rules(),
            existing_player_mindset=existing_player_mindset,
        )

        vote_target = self._decide_player_vote(state, player_id, updated_mindset)

        delta_private = self._create_player_private_state_delta(
            updated_mindset,
            player_context,
            my_word,
        )
        vote_payload = Vote(
            target=vote_target,
            ts=int(datetime.utcnow().timestamp() * 1000),
            phase_id=state["phase_id"],
        )

        self._record_vote_metrics(ctx, vote_target)

        return {
            "current_votes": {player_id: vote_payload},
            "player_private_states": {player_id: delta_private},
        }

    def _record_vote_metrics(
        self,
        ctx: PlayerNodeContext,
        vote_target: str,
    ) -> None:
        metrics_collector.on_vote_cast(
            game_id=ctx.state.get("game_id"),
            round_number=ctx.state["current_round"],
            voter_id=ctx.player_id,
            vote_target=vote_target,
        )

    @staticmethod
    def _decide_player_vote(
        state: GameState,
        player_id: str,
        mindset: PlayerMindset,
    ) -> str:
        suspicions = mindset.suspicions
        strongest_suspicion = max(
            suspicions.items(),
            key=lambda item: getattr(item[1], "confidence", 0),
            default=None,
        )

        if strongest_suspicion:
            suspect_player, suspicion = strongest_suspicion
            if getattr(suspicion, "confidence", 0) > 0.6:
                return suspect_player

        return next_alive_player(state, starting_after=player_id)

    @staticmethod
    def _get_player_context(state: GameState, player_id: str):
        player_context = get_player_context(state, player_id)
        private_state = player_context["private"]
        my_word = private_state.assigned_word
        return player_context, my_word

    @staticmethod
    def _create_player_private_state_delta(
        updated_mindset: PlayerMindset,
        player_context: Dict[str, Any],
        my_word: str,
    ) -> PlayerPrivateState:
        private_state = player_context["private"]
        existing_suspicions = (
            private_state.playerMindset.suspicions
            if private_state.playerMindset
            else {}
        )

        return PlayerPrivateState(
            assigned_word=my_word,
            playerMindset=PlayerMindset(
                self_belief=updated_mindset.self_belief,
                suspicions=merge_probs(existing_suspicions, updated_mindset.suspicions),
            ),
        )
