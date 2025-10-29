import pytest

from src.game.metrics import GameMetrics


@pytest.fixture
def collector(tmp_path, monkeypatch):
    metrics = GameMetrics()
    # Redirect persistence to the temporary directory to avoid polluting logs.
    monkeypatch.setattr(metrics, "_output_dir", tmp_path)
    monkeypatch.setattr(metrics, "_persist_game_summary", lambda summary: None)
    monkeypatch.setattr(metrics, "_persist_overall_metrics", lambda: None)
    metrics.reset()
    return metrics


def test_on_vote_cast_records_vote(collector):
    collector.on_game_start(
        game_id="game-1",
        players=["a", "b"],
        player_roles={"a": "civilian", "b": "spy"},
    )

    collector.on_vote_cast(
        game_id="game-1",
        round_number=1,
        voter_id="a",
        vote_target="b",
    )

    active_game = collector._active_games["game-1"]
    assert len(active_game["vote_records"]) == 1
    record = active_game["vote_records"][0]
    assert record["round_number"] == 1
    assert record["voter_id"] == "a"
    assert record["vote_target"] == "b"
    assert record["voter_role"] == "civilian"
    assert record["target_role"] == "spy"
