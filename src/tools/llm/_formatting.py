"""Shared formatting helpers for LLM prompt construction."""

from __future__ import annotations

from html import escape
from typing import Sequence

from src.game.state import PlayerMindset, Speech


def trim_text_for_prompt(text: str, limit: int = 180) -> str:
    """Normalize whitespace and trim long passages for compact prompts."""

    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "\u2026"


def format_players_xml(players: Sequence[str], alive: Sequence[str], me: str) -> str:
    alive_tags = "".join(f'<player id="{escape(pid)}" status="alive" />' for pid in alive)
    roster_tags = "".join(f'<player id="{escape(pid)}" />' for pid in players)
    return (
        f'<players me="{escape(me)}">'
        f"<alive>{alive_tags or '<none />'}</alive>"
        f"<all>{roster_tags}</all>"
        "</players>"
    )


def format_mindset_xml(player_mindset: PlayerMindset) -> str:
    self_belief = player_mindset.self_belief
    suspicions = player_mindset.suspicions or {}
    suspicions_tags: list[str] = []
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

    ordered = sorted(
        completed_speeches,
        key=lambda speech: (speech.get("round", 0), speech.get("seq", 0)),
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

    segments: list[str] = ["<speech_logs>"]
    current_round: int | None = None

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
