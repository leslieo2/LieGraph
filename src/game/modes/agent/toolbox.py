"""Helper toolbox definitions for agent mode behaviors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from src.tools.llm.inference import llm_update_player_mindset
from src.tools.llm.speech import llm_generate_speech
from ...state import PlayerMindset

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .behaviors import ObservationRecord, PlayerAgentMemory
else:
    from .vote_strategies import (
        vote_align_with_consensus,
        vote_counter_accuser,
        vote_eliminate_prime,
    )


@dataclass(slots=True)
class AgentToolbox:
    """Container for customizable agent tool callables."""

    mindset_updater: Callable[..., PlayerMindset]
    speech_generator: Callable[..., str]
    speech_strategies: Optional[Dict[str, Callable[..., str]]] = None
    vote_strategies: Dict[str, Callable[..., str]] = field(default_factory=dict)
    vote_selector: Optional[Callable[..., str]] = None
    evidence_analyzer: Optional[Callable[[ObservationRecord], Dict[str, float]]] = None
    memory_summarizer: Optional[Callable[[PlayerAgentMemory], str]] = None


def _default_mindset_updater(**kwargs: Any) -> PlayerMindset:
    """Proxy mindset updater that ignores agent-specific kwargs."""

    kwargs.pop("strategy", None)
    kwargs.pop("analysis", None)
    return llm_update_player_mindset(**kwargs)


def _make_strategy_dispatcher(
    strategies: Dict[str, Callable[..., str]],
) -> Callable[..., str]:
    """Create a dispatcher that selects speech generators by strategy key."""

    def _dispatch(**kwargs: Any) -> str:
        dispatch_kwargs = dict(kwargs)
        strategy_key = dispatch_kwargs.pop("strategy", None)
        dispatch_kwargs.pop("analysis", None)

        generator = None
        if strategy_key is not None:
            generator = (
                strategies.get(strategy_key)
                or strategies.get(strategy_key.replace("_", "-"))
                or strategies.get(strategy_key.replace("-", "_"))
            )

        if generator is None:
            generator = strategies.get("default")
        if generator is None and strategies:
            generator = next(iter(strategies.values()))
        if generator is None:
            raise ValueError("Agent toolbox has no speech generators configured")

        return generator(**dispatch_kwargs)

    return _dispatch


def default_toolbox() -> AgentToolbox:
    """Create the default toolbox wired to workflow LLM helpers."""

    strategies: Dict[str, Callable[..., str]] = {
        "default": llm_generate_speech,
        "baseline": llm_generate_speech,
    }

    return AgentToolbox(
        mindset_updater=_default_mindset_updater,
        speech_generator=_make_strategy_dispatcher(strategies),
        speech_strategies=strategies,
        vote_strategies={
            "eliminate-prime": vote_eliminate_prime,
            "consensus": vote_align_with_consensus,
            "defensive": vote_counter_accuser,
        },
    )
