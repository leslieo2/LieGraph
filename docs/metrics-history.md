# Metrics History

Track each batch run and its key metrics to highlight prompt-engineering progress.

## Snapshot Ledger

| Date       | Change Notes                                     | Games Played | Overall Score | Win Balance | Self-ID Avg | Suspicion Trend | Speech Diversity |
|------------|--------------------------------------------------|--------------|---------------|-------------|-------------|-----------------|------------------
| 2025-10-27 | baseline                                         | 5            | 0.5207        | 0.4         | 0.53        | 0.559           | 0.9932           |
| 2025-10-28 | refactor: structure LLM prompts with XML context | 5            | 0.6518        | 0.8         | 0.402       | 0.524           | 0.9874           |
| 2025-10-28 | loosen spy prompts for better balance            | 5            | 0.7116        | 0.8         | 0.569       | 0.5328          | 0.9912           |

### Maintenance Guidelines

- After running `uv run python -m src.game.metrics`, append the new `logs/metrics/overall.json` summary and note any prompt/model/strategy tweaks in the table.
- To keep raw JSON snapshots, copy `logs/metrics/overall.json` to an archive folder (for example `logs/metrics/archive/overall-YYYYMMDD.json`) after each run.
- For per-game detail, refer to the generated files under `logs/metrics/<game_id>.json`.
