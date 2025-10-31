"""
Core strategy coordination for LLM-powered game intelligence.

Coordinates prompt building, context construction, and LLM interaction
for player mindset updates and speech generation.
"""

from typing import Any, List, Dict, Sequence, cast
from venv import logger

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from src.game.state import Speech, PlayerMindset, SelfBelief
from src.game.strategy.context_builder import (
    build_inference_user_context,
    build_speech_user_context,
)
from src.game.strategy.logging_utils import log_self_belief_update
from src.game.strategy.models import PlayerMindsetModel, SelfBeliefModel
from src.game.strategy.prompt_builder import (
    format_inference_system_prompt,
    format_speech_system_prompt,
)
from src.game.strategy.text_utils import sanitize_speech_output


def _to_mindset_model(
    mindset: PlayerMindset | PlayerMindsetModel | None,
) -> PlayerMindsetModel:
    """Convert shared-state mindset data into a Pydantic model."""
    if isinstance(mindset, PlayerMindsetModel):
        return mindset

    if mindset is None:
        return PlayerMindsetModel(
            self_belief=SelfBeliefModel(role="civilian", confidence=0.5),
            suspicions={},
        )

    if hasattr(mindset, "model_dump"):
        return PlayerMindsetModel(**mindset.model_dump())

    return PlayerMindsetModel(**cast(Dict[str, Any], mindset))


def _mindset_model_to_state(model: PlayerMindsetModel) -> PlayerMindset:
    """Convert a Pydantic mindset model into the plain dict state form."""
    return cast(PlayerMindset, model.model_dump())


def llm_update_player_mindset(
    llm_client: Any,
    my_word: str,
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    rules: Dict[str, Any],
    existing_player_mindset: PlayerMindset | None,
) -> PlayerMindset:
    """
    Use LLM to update player's beliefs about their role and suspicions of others.

    Args:
        llm_client: Language model client
        my_word: Player's assigned word
        completed_speeches: History of all speeches
        players: All player IDs
        alive: Currently alive player IDs
        me: Current player's ID
        rules: Game rules dictionary
        existing_player_mindset: Current beliefs and suspicions

    Returns:
        Updated PlayerMindset with new beliefs
    """
    existing_model = _to_mindset_model(existing_player_mindset)
    existing_state = _mindset_model_to_state(existing_model)
    existing_self_belief = existing_state.get(
        "self_belief", {"role": "civilian", "confidence": 0.5}
    )

    # 1. Format the system prompt (instructions)
    system_prompt = format_inference_system_prompt(
        my_word=my_word,
        player_count=len(players),
        spy_count=rules.get("spy_count", 1),
    )

    # 2. Build the user context (structured, dynamic state)
    user_context = build_inference_user_context(
        completed_speeches, players, alive, me, existing_state
    )

    # Create agent with ToolStrategy for structured output so models without
    # native structured output will fall back to tool calling automatically.
    response_format = ToolStrategy(
        schema=PlayerMindsetModel,
        tool_message_content="Player mindset captured.",
    )

    agent = create_agent(
        model=llm_client,
        tools=[],
        response_format=response_format,
    )

    try:
        # Include system prompt in the messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_context),
        ]
        result = agent.invoke({"messages": messages})

        # Extract structured response from agent result
        structured = result.get("structured_response")

        if structured:
            if not isinstance(structured, PlayerMindsetModel):
                structured = PlayerMindsetModel.model_validate(structured)
            new_state = _mindset_model_to_state(structured)
            log_self_belief_update(
                me,
                existing_self_belief,
                new_state.get("self_belief", {"role": "civilian", "confidence": 0.5}),
            )
            return new_state
    except Exception as exc:
        logger.exception("Structured mindset failed: %s", exc)

    # Fallback: LLM failed, preserve previous mindset
    log_self_belief_update(
        me,
        existing_self_belief,
        existing_self_belief,
    )
    return existing_state


def llm_generate_speech(
    llm_client: Any,
    my_word: str,
    self_belief: SelfBelief,
    suspicions: Dict[str, Any],
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    """
    Use LLM to generate a strategic speech based on current beliefs.

    Args:
        llm_client: Language model client
        my_word: Player's assigned word
        self_belief: Current belief about own role
        suspicions: Suspicions about other players (unused but kept for API compatibility)
        completed_speeches: History of all speeches
        me: Current player's ID
        alive: Currently alive player IDs
        current_round: Current game round number

    Returns:
        Generated speech as a single-line string
    """
    system_prompt = format_speech_system_prompt(my_word, self_belief)
    user_context = build_speech_user_context(
        self_belief, completed_speeches, me, alive, current_round
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_context),
    ]

    response = llm_client.invoke(messages)

    raw_text = response.content if hasattr(response, "content") else response
    return sanitize_speech_output(raw_text)
