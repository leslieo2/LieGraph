"""Speech generation helpers shared across game modes."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from src.game.state import SelfBelief, Speech, Suspicion
from src.tools.llm import get_default_llm_client
from src.tools.llm._formatting import format_speeches_xml

__all__ = ["llm_generate_speech"]


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
- Follow the <strategy> tag in the <speech_context> tag and mirror the group’s clarity while masking differences.
- If you sense conflict with the group, emphasize broad categories, shared settings, or emotions instead of specifics.
- Choose 2-3 aspects such as category, purpose, setting, sensory detail, or user that civilians might also mention.
- Mirror the tone and vocabulary other players use.
- Avoid brands, numbers, and rare trivia unless essential.
Reply now with your single-line speech."""


def _determine_clarity(role: str, self_confidence: float, current_round: int) -> tuple[str, str]:
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

    if current_round <= 1:
        return "low", "LOW clarity — broad and neutral foundation"
    if current_round == 2:
        return "medium", "MEDIUM clarity — start introducing gentle differentiators"
    return "high", "HIGH clarity — press with confident, specific traits"


def _build_speech_user_context(
    self_belief: SelfBelief,
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    """Build the dynamic part of the speech prompt as structured XML."""

    clarity_code, clarity_desc = _determine_clarity(
        self_belief.role, self_belief.confidence, current_round
    )

    alive_tags = "".join(f'<player id="{player}" />' for player in alive)
    alive_block = f"<alive_players>{alive_tags or '<none />'}</alive_players>"

    speech_logs_block = format_speeches_xml(completed_speeches)

    return (
        "<speech_context>"
        f'<self role="{self_belief.role}" confidence="{self_belief.confidence:.2f}" />'
        f'<strategy round="{current_round}" clarity="{clarity_code}">{clarity_desc}</strategy>'
        f'<speaker id="{me}" />'
        f'{alive_block}<current_round index="{current_round}" />{speech_logs_block}'
        "<response_guidance>Return exactly one line of speech; avoid emojis, labels, or extra commentary.</response_guidance>"
        "</speech_context>"
    )


_EMOJI_REGEX = re.compile("[\u2600-\u26ff\u2700-\u27bf\U0001f300-\U0001faff]")


def _sanitize_speech_output(text: Any) -> str:
    """Flatten whitespace, drop emojis, and enforce a single-line speech."""

    if text is None:
        return ""

    raw = str(text)
    lines = [line.strip() for line in raw.replace("\r", "").splitlines() if line.strip()]
    candidate = lines[-1] if lines else raw.strip()
    candidate = candidate.replace("\n", " ")
    candidate = _EMOJI_REGEX.sub("", candidate)
    candidate = " ".join(candidate.split())
    return candidate


def _format_speech_system_prompt(my_word: str, self_belief: SelfBelief) -> str:
    is_confident_spy = self_belief.role == "spy" and self_belief.confidence >= 0.7
    template = _SPY_SPEECH_PROMPT_PREFIX if is_confident_spy else _CIVILIAN_SPEECH_PROMPT_PREFIX
    return template.format(my_word=my_word)


def llm_generate_speech(
    *,
    llm_client: Any | None = None,
    my_word: str,
    self_belief: SelfBelief,
    suspicions: Dict[str, Suspicion],
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    system_prompt = _format_speech_system_prompt(my_word, self_belief)
    user_context = _build_speech_user_context(
        self_belief, completed_speeches, me, alive, current_round
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_context),
    ]

    client = llm_client or get_default_llm_client()
    response = client.invoke(messages)

    raw_text = response.content if hasattr(response, "content") else response
    return _sanitize_speech_output(raw_text)
