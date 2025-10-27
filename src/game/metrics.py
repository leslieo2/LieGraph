"""
Metrics collection and scoring utilities for assessing game quality.

This module introduces a lightweight collector that observes the existing game
flow and derives three categories of metrics:
1. Win balance between spies and civilians (target ~50/50 across games)
2. Identification accuracy trends from player mindsets (self and others)
3. Speech diversity based on lexical variety for each utterance

The collector is intentionally decoupled from the game logic so that existing
code only needs to notify it about key events (game start, mindset updates,
speeches, and game end). All heavy lifting—aggregation, trend analysis, and
score computation—lives here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import json
import re
from pathlib import Path

from .state import PlayerMindset


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Return the mean of non-null values or None if nothing is available."""
    filtered = [v for v in values if v is not None]
    return mean(filtered) if filtered else None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a float into [lower, upper]."""
    return max(lower, min(upper, value))


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text for lexical diversity calculation.

    Uses a simple word boundary regex and falls back to character-level tokens
    when the text does not contain alphanumeric segments (works for CJK).
    """
    if not text:
        return []

    tokens = re.findall(r"\w+", text.lower())
    if tokens:
        return tokens

    stripped = text.strip()
    return list(stripped) if stripped else []


@dataclass
class MindsetRecord:
    round_number: int
    phase: str
    player_id: str
    self_accuracy: float
    suspicion_accuracy: Optional[float]


@dataclass
class SpeechRecord:
    round_number: int
    player_id: str
    unique_tokens: int
    total_tokens: int
    diversity: float


BASE_DIR = Path(__file__).resolve().parents[2]


class GameMetrics:
    """Collects per-game data and produces aggregate scores."""

    def __init__(self) -> None:
        self.completed_games: List[Dict[str, Any]] = []
        self.win_counts: Counter[str] = Counter()
        self._current_game: Optional[Dict[str, Any]] = None
        self._output_dir = BASE_DIR / "logs" / "metrics"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Event hooks
    # ------------------------------------------------------------------ #

    def on_game_start(
        self,
        *,
        game_id: str | None,
        players: List[str],
        player_roles: Dict[str, str],
    ) -> None:
        """Initialize collectors for a new game."""
        if not players or not player_roles:
            return

        target_game_id = game_id or f"game-{len(self.completed_games) + 1}"

        if self._current_game:
            active_id = self._current_game.get("game_id")
            if active_id == target_game_id:
                return
            self._finalize_current_game()

        self._current_game = {
            "game_id": target_game_id,
            "players": list(players),
            "roles": dict(player_roles),
            "mindset_records": [],
            "speech_records": [],
            "winner": None,
        }

    def on_player_mindset_update(
        self,
        *,
        game_id: str | None,
        round_number: int,
        phase: str,
        player_id: str,
        mindset: PlayerMindset,
    ) -> None:
        """Record identification accuracy as players update their mindset."""
        game = self._ensure_current(game_id)
        if not game or player_id not in game["roles"]:
            return

        roles = game["roles"]
        records: List[MindsetRecord] = game["mindset_records"]

        self_accuracy = self._accuracy_score(
            actual_role=roles[player_id],
            predicted_role=mindset.self_belief.role,
            confidence=mindset.self_belief.confidence,
        )

        suspicion_scores: List[float] = []
        for other_id, suspicion in mindset.suspicions.items():
            if other_id not in roles:
                continue
            suspicion_scores.append(
                self._accuracy_score(
                    actual_role=roles[other_id],
                    predicted_role=suspicion.role,
                    confidence=suspicion.confidence,
                )
            )

        suspicion_accuracy = (
            sum(suspicion_scores) / len(suspicion_scores)
            if suspicion_scores
            else None
        )

        records.append(
            MindsetRecord(
                round_number=round_number,
                phase=phase,
                player_id=player_id,
                self_accuracy=self_accuracy,
                suspicion_accuracy=suspicion_accuracy,
            )
        )

    def on_speech(
        self,
        *,
        game_id: str | None,
        round_number: int,
        player_id: str,
        content: str,
    ) -> None:
        """Capture lexical diversity for each speech."""
        game = self._ensure_current(game_id)
        if not game:
            return

        tokens = _tokenize(content)
        total = len(tokens)
        unique = len(set(tokens)) if tokens else 0
        diversity = unique / total if total else 0.0

        speeches: List[SpeechRecord] = game["speech_records"]
        speeches.append(
            SpeechRecord(
                round_number=round_number,
                player_id=player_id,
                unique_tokens=unique,
                total_tokens=total,
                diversity=diversity,
            )
        )

    def on_game_end(self, *, game_id: str | None, winner: str | None) -> None:
        """Finalize metrics for the current game."""
        game = self._ensure_current(game_id)
        if not game:
            return

        game["winner"] = winner
        if winner:
            self.win_counts[winner] += 1

        summary = self._summarize_game(game)
        self.completed_games.append(summary)
        self._persist_game_summary(summary)
        self._persist_overall_metrics()
        self._current_game = None

    # ------------------------------------------------------------------ #
    # Public reporting API
    # ------------------------------------------------------------------ #

    def get_overall_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all completed games."""
        total_games = len(self.completed_games)
        civilian_wins = self.win_counts.get("civilians", 0)
        spy_wins = self.win_counts.get("spies", 0)

        win_rate = {
            "civilians": civilian_wins / total_games if total_games else 0.0,
            "spies": spy_wins / total_games if total_games else 0.0,
        }

        win_balance_score = 0.0
        if total_games:
            win_balance_score = 1.0 - abs(win_rate["civilians"] - win_rate["spies"])

        identification = self._aggregate_identification_metrics()
        speech_diversity = self._aggregate_speech_metrics()

        return {
            "games_played": total_games,
            "win_rate": win_rate,
            "win_balance_score": _clamp(win_balance_score),
            "identification": identification,
            "speech_diversity": speech_diversity,
            "game_summaries": list(self.completed_games),
        }

    def compute_quality_score(
        self,
        *,
        method: str = "function",
        llm=None,
    ):
        """
        Compute an overall quality score.

        Args:
            method: "function" (default) for deterministic scoring, or "llm" to
                produce an LLM-evaluated narrative score.
            llm: Optional LLM client following the LangChain Runnable interface.
        """
        summary = self.get_overall_metrics()

        if method == "function":
            return self._compute_functional_score(summary)

        if method == "llm":
            if llm is None:
                raise ValueError("LLM client required when method='llm'.")
            prompt = self._format_summary_for_llm(summary)
            return llm.invoke(prompt)

        raise ValueError(f"Unsupported scoring method: {method}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_current(self, game_id: str | None) -> Optional[Dict[str, Any]]:
        if self._current_game is None:
            return None
        if game_id is None or self._current_game.get("game_id") == game_id:
            return self._current_game
        return None

    @staticmethod
    def _accuracy_score(
        *,
        actual_role: str,
        predicted_role: str,
        confidence: float,
    ) -> float:
        """
        Convert role prediction and confidence into a 0..1 accuracy score.

        Returns confidence when the prediction matches, or the complement when
        incorrect. This rewards calibrated confidence.
        """
        confidence = _clamp(confidence)
        if predicted_role == actual_role:
            return confidence
        return 1.0 - confidence

    def _summarize_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        mindsets: List[MindsetRecord] = game.get("mindset_records", [])
        speeches: List[SpeechRecord] = game.get("speech_records", [])

        round_groups: Dict[int, List[MindsetRecord]] = defaultdict(list)
        for record in mindsets:
            round_groups[record.round_number].append(record)

        per_round = {}
        for round_number, records in sorted(round_groups.items()):
            per_round[round_number] = {
                "self_accuracy": _safe_mean(r.self_accuracy for r in records),
                "suspicion_accuracy": _safe_mean(
                    r.suspicion_accuracy for r in records
                ),
            }

        self_trend = self._trend(per_round, key="self_accuracy")
        suspicion_trend = self._trend(per_round, key="suspicion_accuracy")

        speech_summary = self._summarize_speeches(speeches)

        return {
            "game_id": game.get("game_id"),
            "winner": game.get("winner"),
            "round_metrics": per_round,
            "self_accuracy_trend": self_trend,
            "suspicion_accuracy_trend": suspicion_trend,
            "speech_diversity": speech_summary,
        }

    @staticmethod
    def _trend(round_metrics: Dict[int, Dict[str, Optional[float]]], *, key: str):
        if not round_metrics:
            return None

        ordered_rounds = [r for r in sorted(round_metrics) if round_metrics[r].get(key) is not None]
        if len(ordered_rounds) < 2:
            return None

        first = round_metrics[ordered_rounds[0]][key]
        last = round_metrics[ordered_rounds[-1]][key]
        if first is None or last is None:
            return None
        return last - first

    @staticmethod
    def _summarize_speeches(speeches: List[SpeechRecord]) -> Dict[str, Any]:
        if not speeches:
            return {
                "average_diversity": 0.0,
                "average_unique_tokens": 0.0,
                "average_total_tokens": 0.0,
                "by_player": {},
            }

        by_player: Dict[str, List[SpeechRecord]] = defaultdict(list)
        for record in speeches:
            by_player[record.player_id].append(record)

        def _avg(records: Iterable[SpeechRecord], attr: str) -> float:
            return mean(getattr(r, attr) for r in records)

        per_player = {
            player_id: {
                "average_diversity": _avg(records, "diversity"),
                "average_unique_tokens": _avg(records, "unique_tokens"),
                "average_total_tokens": _avg(records, "total_tokens"),
            }
            for player_id, records in by_player.items()
        }

        return {
            "average_diversity": mean(r.diversity for r in speeches),
            "average_unique_tokens": mean(r.unique_tokens for r in speeches),
            "average_total_tokens": mean(r.total_tokens for r in speeches),
            "by_player": per_player,
        }

    def _aggregate_identification_metrics(self) -> Dict[str, Any]:
        if not self.completed_games:
            return {
                "average_self_accuracy": 0.0,
                "average_suspicion_accuracy": 0.0,
                "self_accuracy_trend": None,
                "suspicion_accuracy_trend": None,
            }

        self_scores = []
        suspicion_scores = []
        self_trends = []
        suspicion_trends = []

        for game in self.completed_games:
            round_metrics = game.get("round_metrics", {})
            if round_metrics:
                self_scores.extend(
                    v.get("self_accuracy")
                    for v in round_metrics.values()
                    if v.get("self_accuracy") is not None
                )
                suspicion_scores.extend(
                    v.get("suspicion_accuracy")
                    for v in round_metrics.values()
                    if v.get("suspicion_accuracy") is not None
                )
            trend = game.get("self_accuracy_trend")
            if trend is not None:
                self_trends.append(trend)
            suspicion_trend = game.get("suspicion_accuracy_trend")
            if suspicion_trend is not None:
                suspicion_trends.append(suspicion_trend)

        return {
            "average_self_accuracy": _safe_mean(self_scores) or 0.0,
            "average_suspicion_accuracy": _safe_mean(suspicion_scores) or 0.0,
            "self_accuracy_trend": _safe_mean(self_trends),
            "suspicion_accuracy_trend": _safe_mean(suspicion_trends),
        }

    def _aggregate_speech_metrics(self) -> Dict[str, Any]:
        if not self.completed_games:
            return {
                "average_diversity": 0.0,
                "average_unique_tokens": 0.0,
                "average_total_tokens": 0.0,
                "by_player": {},
            }

        diversity = []
        unique_tokens = []
        total_tokens = []
        by_player: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for game in self.completed_games:
            speech_summary = game.get("speech_diversity", {})
            global_diversity = speech_summary.get("average_diversity")
            global_unique = speech_summary.get("average_unique_tokens")
            global_total = speech_summary.get("average_total_tokens")
            if global_diversity is not None:
                diversity.append(global_diversity)
            if global_unique is not None:
                unique_tokens.append(global_unique)
            if global_total is not None:
                total_tokens.append(global_total)

            for player_id, stats in speech_summary.get("by_player", {}).items():
                by_player[player_id].append(stats)

        per_player = {
            player_id: {
                "average_diversity": _safe_mean(
                    s.get("average_diversity") for s in stats
                ),
                "average_unique_tokens": _safe_mean(
                    s.get("average_unique_tokens") for s in stats
                ),
                "average_total_tokens": _safe_mean(
                    s.get("average_total_tokens") for s in stats
                ),
            }
            for player_id, stats in by_player.items()
        }

        return {
            "average_diversity": _safe_mean(diversity) or 0.0,
            "average_unique_tokens": _safe_mean(unique_tokens) or 0.0,
            "average_total_tokens": _safe_mean(total_tokens) or 0.0,
            "by_player": per_player,
        }

    @staticmethod
    def _compute_functional_score(summary: Dict[str, Any]) -> Dict[str, float]:
        win_balance = summary.get("win_balance_score", 0.0)
        identification = summary.get("identification", {})
        speech = summary.get("speech_diversity", {})

        identification_trend = identification.get("self_accuracy_trend") or 0.0
        identification_avg = identification.get("average_self_accuracy") or 0.0
        suspicion_trend = identification.get("suspicion_accuracy_trend") or 0.0

        speech_diversity = speech.get("average_diversity") or 0.0

        # Normalize component scores into 0..1 before weighting.
        id_component = _clamp((identification_avg + identification_trend * 0.5))
        suspicion_component = _clamp(0.5 + suspicion_trend / 2)
        speech_component = _clamp(speech_diversity)

        overall = (
            0.45 * win_balance
            + 0.35 * id_component
            + 0.2 * ((speech_component + suspicion_component) / 2)
        )

        return {
            "overall_score": round(overall, 4),
            "win_balance": round(win_balance, 4),
            "identification": round(id_component, 4),
            "suspicion_trend": round(suspicion_component, 4),
            "speech_diversity": round(speech_component, 4),
        }

    @staticmethod
    def _format_summary_for_llm(summary: Dict[str, Any]) -> Dict[str, Any]:
        """Format metrics into an instruction for an LLM reviewer."""
        instructions = (
            "You are evaluating the quality of repeated 'Who Is Spy' games. "
            "Consider whether the competition balance is fair, whether players "
            "improve their ability to identify roles, and whether speeches remain "
            "diverse. Provide a score from 0 to 1 and a short rationale."
        )
        content = {
            "summary": summary,
            "guidance": instructions,
        }
        return {"input": content}

    def _persist_game_summary(self, summary: Dict[str, Any]) -> None:
        game_id = summary.get("game_id") or f"game-{len(self.completed_games)}"
        path = self._output_dir / f"{game_id}.json"
        payload = {"summary": summary}
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def _persist_overall_metrics(self) -> None:
        path = self._output_dir / "overall.json"
        summary = self.get_overall_metrics()
        score = self._compute_functional_score(summary)
        payload = {
            "metrics": summary,
            "quality_score": score,
        }
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def _finalize_current_game(self) -> None:
        """Clean up stale game state when a new game starts unexpectedly."""
        if not self._current_game:
            return
        summary = self._summarize_game(self._current_game)
        self.completed_games.append(summary)
        self._persist_game_summary(summary)
        self._persist_overall_metrics()
        self._current_game = None


# Global collector used by the rest of the codebase.
metrics_collector = GameMetrics()

__all__ = ["GameMetrics", "metrics_collector"]
