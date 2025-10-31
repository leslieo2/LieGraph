import json

from src.game.metrics import GameMetrics, load_saved_game_summaries


def test_load_saved_game_summaries_skips_overall(tmp_path):
    summary = {
        "summary": {
            "game_id": "game-1",
            "winner": "civilians",
            "round_metrics": {},
            "self_accuracy_trend": None,
            "suspicion_accuracy_trend": None,
            "speech_diversity": {
                "average_diversity": 0.0,
                "average_unique_tokens": 0.0,
                "average_total_tokens": 0.0,
                "by_player": {},
            },
        }
    }
    (tmp_path / "game-1.json").write_text(json.dumps(summary), encoding="utf-8")
    # This file should be ignored.
    (tmp_path / "overall.json").write_text("{}", encoding="utf-8")

    loaded = load_saved_game_summaries(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["game_id"] == "game-1"


def test_aggregate_from_summaries():
    summaries = [
        {
            "game_id": "game-1",
            "winner": "civilians",
            "round_metrics": {1: {"self_accuracy": 0.8, "suspicion_accuracy": 0.6}},
            "self_accuracy_trend": 0.1,
            "suspicion_accuracy_trend": 0.05,
            "speech_diversity": {
                "average_diversity": 0.4,
                "average_unique_tokens": 3.0,
                "average_total_tokens": 5.0,
                "by_player": {},
            },
        },
        {
            "game_id": "game-2",
            "winner": "spies",
            "round_metrics": {
                1: {"self_accuracy": 0.6, "suspicion_accuracy": 0.4},
                2: {"self_accuracy": 0.7, "suspicion_accuracy": None},
            },
            "self_accuracy_trend": 0.0,
            "suspicion_accuracy_trend": -0.05,
            "speech_diversity": {
                "average_diversity": 0.5,
                "average_unique_tokens": 4.0,
                "average_total_tokens": 6.0,
                "by_player": {},
            },
        },
    ]

    result = GameMetrics.aggregate_from_summaries(summaries)
    metrics = result["metrics"]
    quality = result["quality_score"]

    assert metrics["games_played"] == 2
    assert metrics["win_rate"] == {"civilians": 0.5, "spies": 0.5}
    assert metrics["win_balance_score"] == 1.0
    identification = metrics["identification"]
    assert identification["average_self_accuracy"] == 0.7
    assert identification["average_suspicion_accuracy"] == 0.5
    assert identification["self_accuracy_trend"] == 0.05
    assert identification["suspicion_accuracy_trend"] == 0.0
    speech = metrics["speech_diversity"]
    assert speech["average_diversity"] == 0.45
    assert speech["average_unique_tokens"] == 3.5
    assert speech["average_total_tokens"] == 5.5

    from pytest import approx

    assert quality["overall_score"] == approx(0.7988, abs=1e-4)
    assert quality["win_balance"] == approx(1.0, abs=1e-4)
    assert quality["identification"] == approx(0.725, abs=1e-4)
    assert quality["suspicion_trend"] == approx(0.5, abs=1e-4)
    assert quality["speech_diversity"] == approx(0.45, abs=1e-4)
