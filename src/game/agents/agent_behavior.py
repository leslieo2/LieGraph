"""Agent-oriented behavior implementations with lightweight memory and strategy logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..config import get_config
from ..llm_strategy import llm_generate_speech, llm_update_player_mindset
from ..metrics import metrics_collector
from ..rules import assign_roles_and_words, calculate_eliminated_player
from ..state import (
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
from .interfaces import (
    BehaviorResult,
    HostBehavior,
    HostNodeContext,
    PlayerBehavior,
    PlayerNodeContext,
)
from .workflow_behaviors import WorkflowHostBehavior, WorkflowPlayerBehavior


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ObservationRecord:
    """Represents an observation captured before the agent makes a decision."""

    round_number: int
    phase: str
    focus: str
    details: Dict[str, Any]
    ts: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class DecisionRecord:
    """Tracks the strategy selected by the agent."""

    round_number: int
    kind: str
    strategy: str
    target: Optional[str] = None
    justification: Optional[str] = None
    ts: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class PlayerAgentMemory:
    """Lightweight memory structure for an agent-controlled player."""

    observations: List[ObservationRecord] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    mindset_history: List[PlayerMindset] = field(default_factory=list)

    def remember_observation(self, record: ObservationRecord) -> None:
        self.observations.append(record)

    def remember_decision(self, record: DecisionRecord) -> None:
        self.decisions.append(record)

    def remember_mindset(self, mindset: PlayerMindset) -> None:
        self.mindset_history.append(mindset)


@dataclass(slots=True)
class HostJournalEntry:
    """Simple journal entry for host agent actions."""

    phase: str
    action: str
    payload: Dict[str, Any]
    ts: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class AgentToolbox:
    """Container for customizable agent tool callables."""

    mindset_updater: Callable[..., PlayerMindset]
    speech_generator: Callable[..., str]
    vote_selector: Optional[Callable[..., str]] = None
    evidence_analyzer: Optional[Callable[[ObservationRecord], Dict[str, float]]] = None
    memory_summarizer: Optional[Callable[[PlayerAgentMemory], str]] = None


def _default_mindset_updater(**kwargs: Any) -> PlayerMindset:
    """Proxy mindset updater that ignores agent-specific kwargs."""

    kwargs.pop("strategy", None)
    kwargs.pop("analysis", None)
    return llm_update_player_mindset(**kwargs)


def _default_speech_generator(**kwargs: Any) -> str:
    """Proxy speech generator that ignores agent-specific kwargs."""

    kwargs.pop("strategy", None)
    kwargs.pop("analysis", None)
    return llm_generate_speech(**kwargs)


def _default_toolbox() -> AgentToolbox:
    """Create the default toolbox wired to workflow LLM helpers."""

    return AgentToolbox(
        mindset_updater=_default_mindset_updater,
        speech_generator=_default_speech_generator,
    )


class AgentPlayerBehavior(WorkflowPlayerBehavior):
    """Agent behavior that extends workflow logic with memory and strategy selection."""

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        toolbox: AgentToolbox | None = None,
    ) -> None:
        super().__init__(llm_client=llm_client)
        self.toolbox = toolbox or _default_toolbox()
        self._memory: Dict[str, PlayerAgentMemory] = {}

    def memory_for(self, player_id: str) -> PlayerAgentMemory:
        """Expose the accumulated memory for a given player."""

        if player_id not in self._memory:
            self._memory[player_id] = PlayerAgentMemory()
        return self._memory[player_id]

    def decide_speech(self, ctx: PlayerNodeContext) -> BehaviorResult:
        if ctx.state["game_phase"] != "speaking":
            return {}
        if ctx.player_id not in alive_players(ctx.state):
            return {}

        player_context, my_word = self._get_player_context(ctx.state, ctx.player_id)
        private_state = player_context["private"]
        memory = self.memory_for(ctx.player_id)

        observation = self._observe(ctx, focus="speech")
        memory.remember_observation(observation)

        updated_mindset = self._evaluate_mindset(
            ctx,
            observation=observation,
            existing=private_state.playerMindset,
            assigned_word=my_word,
        )
        memory.remember_mindset(updated_mindset)

        strategy = self._choose_strategy(
            ctx.state,
            player_id=ctx.player_id,
            mindset=updated_mindset,
            focus="speech",
        )

        decision = DecisionRecord(
            round_number=ctx.state["current_round"],
            kind="speech",
            strategy=strategy,
        )
        memory.remember_decision(decision)

        speech_text = self.toolbox.speech_generator(
            llm_client=self.llm_client,
            my_word=my_word,
            self_belief=updated_mindset.self_belief,
            suspicions=updated_mindset.suspicions,
            completed_speeches=ctx.state["completed_speeches"],
            me=ctx.player_id,
            alive=alive_players(ctx.state),
            current_round=ctx.state["current_round"],
            strategy=strategy,
            analysis=self._extract_analysis(memory),
        )

        metrics_collector.on_player_mindset_update(
            game_id=ctx.state.get("game_id"),
            round_number=ctx.state["current_round"],
            phase=ctx.state["game_phase"],
            player_id=ctx.player_id,
            mindset=updated_mindset,
        )
        metrics_collector.on_speech(
            game_id=ctx.state.get("game_id"),
            round_number=ctx.state["current_round"],
            player_id=ctx.player_id,
            content=speech_text,
        )

        speech_record: Speech = create_speech_record(
            state=ctx.state,
            player_id=ctx.player_id,
            content=speech_text,
        )

        delta_private = self._create_player_private_state_delta(
            updated_mindset,
            player_context,
            my_word,
        )

        return {
            "completed_speeches": [speech_record],
            "player_private_states": {ctx.player_id: delta_private},
        }

    def decide_vote(self, ctx: PlayerNodeContext) -> BehaviorResult:
        if ctx.state["game_phase"] != "voting":
            return {}
        if ctx.player_id not in alive_players(ctx.state):
            return {}

        player_context, my_word = self._get_player_context(ctx.state, ctx.player_id)
        private_state = player_context["private"]
        memory = self.memory_for(ctx.player_id)

        observation = self._observe(ctx, focus="vote")
        memory.remember_observation(observation)

        updated_mindset = self._evaluate_mindset(
            ctx,
            observation=observation,
            existing=private_state.playerMindset,
            assigned_word=my_word,
        )
        memory.remember_mindset(updated_mindset)

        strategy = self._choose_strategy(
            ctx.state,
            player_id=ctx.player_id,
            mindset=updated_mindset,
            focus="vote",
        )

        vote_target = self._select_vote_target(
            ctx.state,
            player_id=ctx.player_id,
            mindset=updated_mindset,
            strategy=strategy,
        )

        decision = DecisionRecord(
            round_number=ctx.state["current_round"],
            kind="vote",
            strategy=strategy,
            target=vote_target,
        )
        memory.remember_decision(decision)

        metrics_collector.on_player_mindset_update(
            game_id=ctx.state.get("game_id"),
            round_number=ctx.state["current_round"],
            phase=ctx.state["game_phase"],
            player_id=ctx.player_id,
            mindset=updated_mindset,
        )

        delta_private = self._create_player_private_state_delta(
            updated_mindset,
            player_context,
            my_word,
        )
        vote_payload = Vote(
            target=vote_target,
            ts=int(_utc_now().timestamp() * 1000),
            phase_id=ctx.state["phase_id"],
        )

        return {
            "current_votes": {ctx.player_id: vote_payload},
            "player_private_states": {ctx.player_id: delta_private},
        }

    def _observe(self, ctx: PlayerNodeContext, *, focus: str) -> ObservationRecord:
        """Gather observable context for the current decision."""

        state = ctx.state
        details = {
            "alive": list(alive_players(state)),
            "completed_speeches": [
                speech
                for speech in state.get("completed_speeches", [])
                if speech.get("round") == state["current_round"]
            ],
            "votes": self._valid_votes(state),
        }
        return ObservationRecord(
            round_number=state["current_round"],
            phase=state["game_phase"],
            focus=focus,
            details=details,
        )

    def _evaluate_mindset(
        self,
        ctx: PlayerNodeContext,
        *,
        observation: ObservationRecord,
        existing: PlayerMindset | None,
        assigned_word: str,
    ) -> PlayerMindset:
        """Update the player's mindset using configured tooling."""

        config = get_config()
        updater = ctx.extras.get(
            "llm_update_player_mindset", self.toolbox.mindset_updater
        )
        analysis = None
        if self.toolbox.evidence_analyzer:
            analysis = self.toolbox.evidence_analyzer(observation)

        kwargs: Dict[str, Any] = {
            "llm_client": self.llm_client,
            "my_word": assigned_word,
            "completed_speeches": ctx.state["completed_speeches"],
            "players": ctx.state["players"],
            "alive": alive_players(ctx.state),
            "me": ctx.player_id,
            "rules": config.get_game_rules(),
            "existing_player_mindset": existing,
        }
        if analysis is not None:
            kwargs["analysis"] = analysis
        kwargs["strategy"] = observation.focus

        try:
            return updater(**kwargs)
        except TypeError:
            kwargs.pop("analysis", None)
            kwargs.pop("strategy", None)
            return updater(**kwargs)

    def _choose_strategy(
        self,
        state: GameState,
        *,
        player_id: str,
        mindset: PlayerMindset,
        focus: str,
    ) -> str:
        """Select a strategy label using simple heuristics."""

        suspicion_strength = self._max_suspicion(mindset.suspicions.values())
        if focus == "speech":
            if (
                mindset.self_belief.role == "spy"
                and mindset.self_belief.confidence >= 0.7
            ):
                return "blend-in"
            if suspicion_strength and suspicion_strength > 0.7:
                return "press-lead"
            if state["current_round"] == 1:
                return "seed-context"
            return "reinforce"

        # Voting focus
        if suspicion_strength and suspicion_strength > 0.8:
            return "eliminate-prime"
        if mindset.self_belief.confidence < 0.5:
            return "defensive"
        return "consensus"

    def _select_vote_target(
        self,
        state: GameState,
        *,
        player_id: str,
        mindset: PlayerMindset,
        strategy: str,
    ) -> str:
        selector = self.toolbox.vote_selector
        if selector is not None:
            return selector(
                state=state,
                player_id=player_id,
                mindset=mindset,
                strategy=strategy,
            )
        return self._decide_player_vote(state, player_id, mindset)

    def _extract_analysis(self, memory: PlayerAgentMemory) -> Optional[str]:
        if not self.toolbox.memory_summarizer:
            return None
        return self.toolbox.memory_summarizer(memory)

    @staticmethod
    def _max_suspicion(suspicions: Iterable[Any]) -> Optional[float]:
        confidences = []
        for suspicion in suspicions:
            confidence = getattr(suspicion, "confidence", None)
            if confidence is not None:
                confidences.append(float(confidence))
        return max(confidences) if confidences else None

    @staticmethod
    def _valid_votes(state: GameState) -> Dict[str, Vote]:
        return {
            voter: vote
            for voter, vote in state.get("current_votes", {}).items()
            if getattr(vote, "phase_id", None) == state.get("phase_id")
        }

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

    @staticmethod
    def _get_player_context(state: GameState, player_id: str):
        player_context = get_player_context(state, player_id)
        private_state = player_context["private"]
        my_word = private_state.assigned_word
        return player_context, my_word


class AgentHostBehavior(WorkflowHostBehavior):
    """Host behavior that augments workflow logic with decision journaling."""

    def __init__(self) -> None:
        super().__init__()
        self.journal: List[HostJournalEntry] = []

    def setup(self, ctx: HostNodeContext) -> BehaviorResult:
        entry = HostJournalEntry(
            phase="setup",
            action="assign_roles",
            payload={"players": list(ctx.state.get("players", []))},
        )
        self.journal.append(entry)
        return super().setup(ctx)

    def stage_switch(self, ctx: HostNodeContext) -> BehaviorResult:
        state = ctx.state
        next_player = (
            next_alive_player(state) if state["game_phase"] == "speaking" else None
        )
        entry = HostJournalEntry(
            phase=state["game_phase"],
            action="stage_switch",
            payload={"next_player": next_player},
        )
        self.journal.append(entry)
        return super().stage_switch(ctx)

    def resolve_round(self, ctx: HostNodeContext) -> BehaviorResult:
        state = ctx.state
        eliminated_player = calculate_eliminated_player(state)

        entry = HostJournalEntry(
            phase="voting",
            action="resolve_round",
            payload={
                "current_round": state["current_round"],
                "votes": dict(state.get("current_votes", {})),
                "eliminated_player": eliminated_player,
            },
        )
        self.journal.append(entry)

        result = super().resolve_round(ctx)

        if result.get("winner"):
            self.journal.append(
                HostJournalEntry(
                    phase="result",
                    action="announce_winner",
                    payload={"winner": result["winner"]},
                )
            )
        return result


def build_agent_game_state(
    state: GameState, player_list: Optional[List[str]] = None
) -> BehaviorResult:
    """Utility to produce initial game setup using agent host logic."""

    players = player_list or state.get("players")
    if not players:
        config = get_config()
        players = config.generate_player_names()

    host_private_state = state.get("host_private_state", {})
    assignments = assign_roles_and_words(
        players,
        host_private_state=host_private_state,
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
        "players": players,
    }
    game_setup_state["phase_id"] = generate_phase_id(game_setup_state)
    return game_setup_state
