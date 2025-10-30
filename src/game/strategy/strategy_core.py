"""
Core strategy coordination for LLM-powered game intelligence.

Coordinates prompt building, context construction, and LLM interaction
for player mindset updates and speech generation.
"""

from typing import Any, List, Dict, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from src.game.state import Speech, PlayerMindset, SelfBelief
from src.game.strategy.logging_utils import log_self_belief_update
from src.game.strategy.prompt_builder import (
    format_inference_system_prompt,
    format_speech_system_prompt,
)
from src.game.strategy.context_builder import (
    build_inference_user_context,
    build_speech_user_context,
)
from src.game.strategy.text_utils import sanitize_speech_output


def llm_update_player_mindset(
    llm_client: Any,
    my_word: str,
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    rules: Dict[str, Any],
    existing_player_mindset: PlayerMindset,
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
    existing_self_belief = existing_player_mindset.self_belief

    # 1. Format the system prompt (instructions)
    system_prompt = format_inference_system_prompt(
        my_word=my_word,
        player_count=len(players),
        spy_count=rules.get("spy_count", 1),
    )

    # 2. Build the user context (structured, dynamic state)
    user_context = build_inference_user_context(
        completed_speeches, players, alive, me, existing_player_mindset
    )

    # Create agent with structured output using ToolStrategy
    agent = create_agent(
        model=llm_client,
        tools=[],  # No additional tools needed for structured output
        response_format=ToolStrategy(PlayerMindset),
    )

    # Invoke the agent with the prompts
    result = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ]
        }
    )

    # Extract structured response from result
    if result.get("structured_response"):
        new_mindset = result["structured_response"]
        log_self_belief_update(me, existing_self_belief, new_mindset.self_belief)
        return new_mindset

    # Fallback: LLM failed, preserve previous mindset
    log_self_belief_update(
        me, existing_self_belief, existing_player_mindset.self_belief
    )
    return existing_player_mindset


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
