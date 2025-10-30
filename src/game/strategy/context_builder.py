"""
Context builders for converting game state to structured LLM input.

Provides XML-based context formatting for inference and speech generation.
"""

from html import escape
from typing import List, Sequence, Dict

from src.game.state import Speech, PlayerMindset, SelfBelief, Suspicion
from src.game.strategy.prompt_builder import determine_clarity


def trim_text_for_prompt(text: str, limit: int = 180) -> str:
    """Normalize whitespace and trim long passages for prompt friendliness."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def format_players_xml(players: Sequence[str], alive: Sequence[str], me: str) -> str:
    """Format player lists as XML."""
    alive_tags = "".join(
        f'<player id="{escape(pid)}" status="alive" />' for pid in alive
    )
    roster_tags = "".join(f'<player id="{escape(pid)}" />' for pid in players)
    return (
        f'<players me="{escape(me)}">'
        f"<alive>{alive_tags or '<none />'}</alive>"
        f"<all>{roster_tags}</all>"
        "</players>"
    )


def format_mindset_xml(player_mindset: PlayerMindset) -> str:
    """Format player mindset (beliefs and suspicions) as XML."""
    self_belief = player_mindset.self_belief
    suspicions = player_mindset.suspicions or {}
    suspicions_tags = []
    for pid, suspicion in suspicions.items():
        trimmed_reason = trim_text_for_prompt(suspicion.reason, limit=160)
        suspicions_tags.append(
            (
                f'<suspicion target="{escape(pid)}" '
                f'role="{escape(suspicion.role)}" '
                f'confidence="{suspicion.confidence:.2f}">'
                f"{escape(trimmed_reason)}"
                "</suspicion>"
            )
        )

    suspicions_block = "".join(suspicions_tags) or "<none />"
    return (
        f'<mindset self_role="{escape(self_belief.role)}" '
        f'self_confidence="{self_belief.confidence:.2f}">'
        f"<suspicions>{suspicions_block}</suspicions>"
        "</mindset>"
    )


def format_speeches_xml(
    completed_speeches: Sequence[Speech],
    rounds_to_keep: int | None = None,
    max_entries: int | None = None,
) -> str:
    """Render speeches in chronological order with optional trimming."""
    if not completed_speeches:
        return "<speech_logs />"

    # Assume completed_speeches is already in chronological order
    if rounds_to_keep is not None:
        round_ids = sorted({speech.get("round", 0) for speech in completed_speeches})
        selected_rounds = set(round_ids[-rounds_to_keep:])
        filtered = [
            speech
            for speech in completed_speeches
            if speech.get("round", 0) in selected_rounds
        ]
    else:
        filtered = completed_speeches

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
                f"{escape(trim_text_for_prompt(speech.get("content", ""), limit=140))}"
                "</speech>"
            )
        )

    if current_round is not None:
        segments.append("</round>")

    segments.append("</speech_logs>")
    return "".join(segments)


def build_inference_user_context(
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    existing_player_mindset: PlayerMindset,
) -> str:
    """Builds the dynamic context for inference (belief update)."""
    players_xml = format_players_xml(players, alive, me)
    mindset_xml = format_mindset_xml(existing_player_mindset)
    speeches_xml = format_speeches_xml(completed_speeches)

    return (
        "<inference_context>"
        f"{players_xml}{mindset_xml}{speeches_xml}"
        "<response_guidance>Use the PlayerMindset tool only; do not provide prose or explanations.</response_guidance>"
        "</inference_context>"
    )


def build_speech_user_context(
    self_belief: SelfBelief,
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    """Builds the dynamic context for speech generation."""
    self_role = self_belief.role
    self_confidence = self_belief.confidence

    clarity_code, clarity_desc = determine_clarity(
        self_role, self_confidence, current_round
    )

    alive_tags = "".join(f'<player id="{escape(pid)}" />' for pid in alive)
    alive_block = f"<alive_players>{alive_tags or '<none />'}</alive_players>"

    speech_logs_block = format_speeches_xml(completed_speeches)

    return (
        "<speech_context>"
        f'<self role="{escape(self_role)}" confidence="{self_confidence:.2f}" />'
        f'<strategy round="{current_round}" clarity="{clarity_code}">{escape(clarity_desc)}</strategy>'
        f'<speaker id="{escape(me)}" />'
        f'{alive_block}<current_round index="{current_round}" />{speech_logs_block}'
        "<response_guidance>Return exactly one line of speech; avoid emojis, labels, or extra commentary.</response_guidance>"
        "</speech_context>"
    )
