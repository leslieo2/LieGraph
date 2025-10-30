"""
Strategic speech generation tools.

This module provides tools for generating player speeches
based on their mindset, confidence, and game state.

Access Control:
- Players can only access their own private state
- Players can access public game state
"""

import re
from typing import Any, Sequence, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from src.game.state import (
    GameState,
    Speech,
    SelfBelief,
    Suspicion,
    create_speech_record,
)
from html import escape


def _determine_clarity(
    role: str, self_confidence, current_round: int
) -> tuple[str, str]:
    """Return role-aware clarity code and description for the current round."""
    if role == "spy" and self_confidence > 0.5:
        if current_round <= 2:
            return (
                "low",
                "LOW clarity — stay broad to blend with civilians",
            )
        if current_round <= 4:
            return (
                "medium",
                "MEDIUM clarity — add safe overlaps without exposing differences",
            )
        return (
            "medium",
            "MEDIUM clarity — stay measured while matching the group's detail level",
        )

    # Civilian defaults
    if current_round <= 1:
        return "low", "LOW clarity — broad and neutral foundation"
    if current_round == 2:
        return "medium", "MEDIUM clarity — start introducing gentle differentiators"
    return "high", "HIGH clarity — press with confident, specific traits"


def _trim_text_for_prompt(text: str, limit: int = 180) -> str:
    """Normalize whitespace and trim long passages for prompt friendliness."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _format_speeches_xml(
    completed_speeches: Sequence[Speech],
    rounds_to_keep: int | None = None,
    max_entries: int | None = None,
) -> str:
    """Render speeches in chronological order with optional trimming."""
    if not completed_speeches:
        return "<speech_logs />"

    ordered = sorted(
        completed_speeches,
        key=lambda s: (s.get("round", 0), s.get("seq", 0)),
    )

    if rounds_to_keep is not None:
        round_ids = sorted({speech.get("round", 0) for speech in ordered})
        selected_rounds = set(round_ids[-rounds_to_keep:])
        filtered = [
            speech for speech in ordered if speech.get("round", 0) in selected_rounds
        ]
    else:
        filtered = ordered

    if max_entries is not None and len(filtered) > max_entries:
        filtered = filtered[-max_entries:]

    segments = ["<speech_logs>"]
    current_round = None

    for speech in filtered:
        round_index = speech.get("round", 0)
        if round_index != current_round:
            if current_round is not None:
                segments.append("</round>")
            segments.append(f'<round index="{round_index}">')
            current_round = round_index

        segments.append(
            (
                f'<speech seq="{speech.get("seq", 0)}" '
                f'player="{escape(speech.get("player_id", "unknown"))}">'
                f"{escape(_trim_text_for_prompt(speech.get("content", ""), limit=140))}"
                "</speech>"
            )
        )

    if current_round is not None:
        segments.append("</round>")

    segments.append("</speech_logs>")
    return "".join(segments)


def _build_speech_user_context(
    self_belief: SelfBelief,
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    """Builds the dynamic part of the speech prompt as structured XML."""
    self_role = self_belief.role
    self_confidence = self_belief.confidence

    clarity_code, clarity_desc = _determine_clarity(
        self_role, self_confidence, current_round
    )

    alive_tags = "".join(f'<player id="{escape(pid)}" />' for pid in alive)
    alive_block = f"<alive_players>{alive_tags or '<none />'}</alive_players>"

    speech_logs_block = _format_speeches_xml(completed_speeches)

    return (
        "<speech_context>"
        f'<self role="{escape(self_role)}" confidence="{self_confidence:.2f}" />'
        f'<strategy round="{current_round}" clarity="{clarity_code}">{escape(clarity_desc)}</strategy>'
        f'<speaker id="{escape(me)}" />'
        f'{alive_block}<current_round index="{current_round}" />{speech_logs_block}'
        "<response_guidance>Return exactly one line of speech; avoid emojis, labels, or extra commentary.</response_guidance>"
        "</speech_context>"
    )


_CIVILIAN_SPEECH_PROMPT_PREFIX = """You are a civilian player in the party game "Who is the Spy". Your secret word is "{my_word}" and it is your turn to speak.
Goal: Share one truthful clue that helps fellow civilians test the group without giving away the exact word.
Must:
- Reply in the same language as "{my_word}".
- Output exactly one line of plain text; no labels, emojis, quotes, or meta reasoning.
- Tell the truth about your word; do not say the word itself or obvious synonyms.
- Do not mention roles, probabilities, mechanics, questions, accusations, or player names.
- Avoid repeating another player's description this round.
- Stay concise: 18-35 characters for Chinese/Japanese/Korean, otherwise 20-40 words.
Guide:
- Follow the <strategy> tag in the <speech_context> tag to match the desired clarity for this turn.
- Use the confidence value in the <self> tag to decide how bold to be: higher confidence supports sharper differentiators, lower confidence favors safer overlaps.
- Choose 2-3 aspects such as category, purpose, setting, sensory detail, or user.
- Mirror the tone and vocabulary other players use.
- Skip brands, numbers, and rare trivia unless essential.
Reply now with your single-line speech."""

_SPY_SPEECH_PROMPT_PREFIX = """You are the spy in the party game "Who is the Spy". Your secret word is "{my_word}" and it is your turn to speak.
Goal: Blend in by giving a plausible clue that could also fit the civilians' word while safely disguising your own.
Must:
- Reply in the same language as "{my_word}".
- Output exactly one line of plain text; no labels, emojis, quotes, or meta reasoning.
- Prioritize overlap with likely civilian clues; you may soften or generalize details to avoid exposing your unique angle.
- Do not mention roles, probabilities, mechanics, questions, accusations, or player names.
- Avoid repeating another player's description this round.
- Stay concise: 18-35 characters for Chinese/Japanese/Korean, otherwise 20-40 words.
Guide:
- Follow the <strategy> tag in the <speech_context> tag and mirror the group's clarity while masking differences.
- If you sense conflict with the group, emphasize broad categories, shared settings, or emotions instead of specifics.
- Choose 2-3 aspects such as category, purpose, setting, sensory detail, or user that civilians might also mention.
- Mirror the tone and vocabulary other players use.
- Avoid brands, numbers, and rare trivia unless essential.
Reply now with your single-line speech."""


_EMOJI_REGEX = re.compile("[\u2600-\u26ff\u2700-\u27bf\U0001f300-\U0001faff]")


def _sanitize_speech_output(text: Any) -> str:
    """Flatten whitespace, drop emojis, and enforce a single-line speech."""
    if text is None:
        return ""

    raw = str(text)
    lines = [
        line.strip() for line in raw.replace("\r", "").splitlines() if line.strip()
    ]
    candidate = lines[-1] if lines else raw.strip()
    candidate = candidate.replace("\n", " ")
    candidate = _EMOJI_REGEX.sub("", candidate)
    candidate = " ".join(candidate.split())
    return candidate


def _format_speech_system_prompt(my_word: str, self_belief: SelfBelief) -> str:
    """Select the civilian or spy speech prompt based on calibrated confidence."""
    is_confident_spy = self_belief.role == "spy" and self_belief.confidence >= 0.7
    if is_confident_spy:
        template = _SPY_SPEECH_PROMPT_PREFIX
    else:
        template = _CIVILIAN_SPEECH_PROMPT_PREFIX
    return template.format(my_word=my_word)


def generate_speech_tool(
    state: GameState,
    player_id: str,
) -> Dict[str, Any]:
    """
    Generate strategic speech for a player.

    This tool generates speech based on:
    - Player's current mindset (self-belief and suspicions)
    - Assigned word
    - Current game phase and round
    - Previous speeches

    Access Control:
    - Can only access player's own private state (player_private_states[player_id])
    - Can access public game state (completed_speeches, current_round, etc.)

    Args:
        state: Current GameState
        player_id: ID of the player generating speech

    Returns:
        Dictionary with new speech added to completed_speeches
    """

    # Access control: validate player_id exists
    if player_id not in state.get("player_private_states", {}):
        raise ValueError(f"Invalid player_id: {player_id}")

    # Check if it's speaking phase
    if state["game_phase"] != "speaking":
        return {}

    # Get player's private state (allowed)
    player_private = state["player_private_states"][player_id]
    my_word = player_private.assigned_word
    player_mindset = player_private.playerMindset

    # Get public game state (allowed)
    completed_speeches = state.get("completed_speeches", [])
    current_round = state["current_round"]

    # Get alive players
    eliminated = set(state.get("eliminated_players", []))
    alive = [p for p in state["players"] if p not in eliminated]

    # Build prompt
    system_prompt = _format_speech_system_prompt(my_word, player_mindset.self_belief)
    user_context = _build_speech_user_context(
        player_mindset.self_belief, completed_speeches, player_id, alive, current_round
    )

    # Call LLM
    from src.tools.llm import create_llm

    llm_client = create_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_context),
    ]

    response = llm_client.invoke(messages)
    raw_text = response.content if hasattr(response, "content") else response
    speech_text = _sanitize_speech_output(raw_text)

    # Create speech record
    speech_record = create_speech_record(state, player_id, speech_text)

    # Return update dictionary
    return {"completed_speeches": [speech_record]}


def analyze_speech_consistency(
    state: GameState,
    player_id: str,
    target_player_id: str,
) -> Dict[str, Any]:
    """
    Analyze speech consistency of a target player.

    This tool analyzes whether target's speeches align with group consensus
    or show outlier patterns.

    Access Control:
    - Can only access public speeches
    - Cannot access any private states

    Args:
        state: Current GameState
        player_id: ID of the player performing analysis
        target_player_id: ID of the player to analyze

    Returns:
        Analysis results with consistency assessment
    """

    # Access control: validate both player IDs
    if player_id not in state.get("player_private_states", {}):
        raise ValueError(f"Invalid player_id: {player_id}")
    if target_player_id not in state["players"]:
        raise ValueError(f"Invalid target_player_id: {target_player_id}")

    # Get all speeches (public data)
    completed_speeches = state.get("completed_speeches", [])

    # Get target player's speeches
    target_speeches = [
        s for s in completed_speeches if s.get("player_id") == target_player_id
    ]

    # Get all other speeches for comparison
    other_speeches = [
        s for s in completed_speeches if s.get("player_id") != target_player_id
    ]

    # Simple analysis
    analysis = {
        "target_player_id": target_player_id,
        "speech_count": len(target_speeches),
        "target_speeches": [s.get("content", "") for s in target_speeches],
        "other_speech_count": len(other_speeches),
        # Placeholder metrics - could be enhanced
        "avg_speech_length": sum(len(s.get("content", "")) for s in target_speeches)
        / max(len(target_speeches), 1),
        "is_consistent": True,  # Placeholder
    }

    return analysis
