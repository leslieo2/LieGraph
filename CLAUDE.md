# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LieGraph is an AI-powered implementation of the social deduction game "Who Is Spy" built with LangGraph. It features autonomous AI agents that use LLM reasoning to find the spy among them.

- **Main Language**: Python 3.12+
- **Core Framework**: LangGraph for workflow orchestration
- **AI Integration**: LangChain with structured LLM outputs
- **Frontend**: React 19.2 with LangGraph SDK

## Development Commands

### Initial Setup
```bash
# Install Python dependencies (uses uv package manager)
uv sync

# Create .env file from template
cp .env.template .env
# Edit .env with your API keys (OpenAI, DeepSeek, or OpenRouter)

# Install frontend dependencies
cd ui-web/frontend && npm install
```

### Running the Application
```bash
# Terminal 1: Start LangGraph backend (from project root)
langgraph dev --config langgraph.json --port 8124 --allow-blocking

# Terminal 2: Start React frontend (from ui-web/frontend)
npm start

# Access UI at: http://localhost:3000
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_game_rules.py -v

# Run specific test
python -m pytest tests/test_game_rules.py::test_assign_roles -v
```

### Linting/Formatting
```bash
# Format Python code
black src/ tests/
```

## Architecture Overview

### LangGraph Workflow Flow
```
START → host_setup → host_stage_switch
                        ↓
                    speaking phase (sequential player_speech nodes)
                        ↓
                    voting phase (concurrent player_vote nodes)
                        ↓
                    check_votes_and_transition
                        ↓
                    host_result → (continue or END)
```

### Key Architectural Patterns

1. **State Management**: TypedDict-based GameState with private state separation
   - `GameState`: Shared public state (speech history, votes, game status)
   - `HostState`: Private host mindset (invariant after setup)
   - `PlayerState`: Private player mindsets (evolving beliefs about identities)

2. **Concurrent Voting**: Multiple players vote in parallel using LangGraph reducers
   - Reducer: `merge_votes` handles timestamp-based conflict resolution
   - Each vote node is independent but writes to shared state

3. **AI Strategy System** (`src/game/strategy/`):
   - `strategy_core.py`: Main LLM coordination
   - `builders/`: Context and prompt builders for speech/voting/inference
   - `llm_schemas.py`: Pydantic models for structured LLM outputs

4. **Agent Tools** (`src/game/agent_tools/`):
   - `speech_tools.py`: Structured reasoning for speech generation
   - `vote_tools.py`: Evidence-based voting decisions
   - Uses TrustCall for reliable structured output extraction

5 **Reducers**: State conflict resolution functions
   - `merge_private_states`: Combine incremental mindset updates
   - `merge_votes`: Handle concurrent vote submissions
   - Use `add` for append-only collections (speeches, votes)

6. **Conditional Routing**: `src/game/graph.py` uses dynamic edge routing
   - `host_stage_switch`: Routes between speaking and voting phases
   - Checks `current_speaker` and vote readiness
   - Returns edge names for graph transitions

7. **PyDict Structured Output**: Uses dict exports for serialization
   - GameState uses `PyDict` export methods for LangGraph checkpoints
   - Ensure all Pydantic models in state have proper serialization

8 **Private State Updates**: Player/hod nodes return private state deltas
   - Add "_" prefix: `{player_name: PlayerState}` → `{"_" + player_name: PlayerState}`
   - Host returns `f"_{HOST_NAME}": HostState`
   - Graph middleware merges private states using configured reducers

9. **Channel Configuration**: Define channels for each node in graph
   - Player nodes: `"_" + player_name` channels
   - Host node: `"_" + HOST_NAME` channel
   - Channels must match reducer keys

## Configuration

**LLM Configuration** (`.env`):
- Supports OpenAI, DeepSeek, and OpenRouter providers
- Set provider-specific API keys and model names
- Example models: `gpt-4o-mini`, `deepseek-chat`, `anthropic/claude-sonnet-4.5`

**Game Configuration** (`config.yaml`):
- `player_count`: Number of players (3-8)
- `vocabulary`: Word pairs for civilian/spy assignments
- `player_names`: Pool of available player names
- `metrics.enabled`: Toggle metrics collection on/off

## Testing Strategy

**Test Coverage** (50 tests across 6 modules):
- `test_game_rules.py`: Core game logic, role assignment, win conditions
- `test_state.py`: State management and reducer functions
- `test_host_nodes.py`: Host node behavior and phase transitions
- `test_player_nodes.py`: Player speech and voting nodes
- `test_llm_strategy.py`: AI strategy builders and prompt generation
- `agents/test_speech_tools.py`: LLM tool behavior and structured outputs

**Key Testing Patterns**:
- Use fixtures for common GameState configurations
- Mock LLM responses for deterministic AI behavior tests
- Test both sequential (speaking) and concurrent (voting) nodes
- Verify private state updates and mindset evolution

## Metrics and Quality Tracking

**Built-in Metrics** (`src/game/metrics.py`):
- Win balance tracking (civilian vs spy win rates)
- Identification accuracy (role inference quality)
- Speech diversity (lexical variety measurement)
- Auto-saves to `logs/metrics/{game_id}.json`
- Overall summary at `logs/metrics/overall.json`

**Quality Scoring**:
```python
from src.game.dependencies import build_dependencies

deps = build_dependencies()
collector = deps.metrics

# Get quality score
deterministic_score = collector.compute_quality_score()

# Or use LLM-based evaluation
llm_score = collector.compute_quality_score(method="llm", llm=client)
```

**Metrics History**: Track prompts and configurations in `docs/metrics-history.md`

## Common Development Tasks

### Adding a New Game Phase
1. Add node function to `src/game/nodes/`
2. Register node in graph with `graph.add_node(node_name, node_function)`
3. Add conditional routing logic in transition nodes
4. Update state types if adding new fields

### Modifying AI Strategy
1. Update prompt builders in `src/game/strategy/builders/`
2. Modify Pydantic schemas in `src/game/strategy/llm_schemas.py` if changing output structure
3. Adjust strategy coordination in `src/game/strategy/strategy_core.py`
4. Test with `pytest tests/test_llm_strategy.py`

### Debugging Game Flow
1. Enable LangSmith tracing: `LANGSMITH_TRACING=true` in `.env`
2. Check LangGraph Studio at `http://localhost:8123`
3. Review game logs in `logs/metrics/`
4. Use `print()` in nodes to debug state (visible in LangGraph Studio traces)

### Adding New Metrics
1. Add metric collection hooks in `src/game/metrics.py`
2. Update quality scoring computation
3. Add metric tests in `tests/test_metrics_history.py`
4. Document in `docs/metrics-history.md`

### Working with Player-Specific Hooks (callbacks) for Metrics
When implementing player-specific behaviors that need to track metrics per player:
- Access the injected collector via the dependencies bundle (e.g., the `metrics` argument supplied to LangGraph nodes).
- Use the `metrics.on_player_speech()` and related hooks within player speech/vote nodes to collect lexical diversity and voting pattern data.
- Metrics collection respects the `metrics.enabled` flag in `config.yaml` and will be no-ops when metrics are disabled.

## LangGraph Development Notes

**Checkpointing**: State is automatically checkpointed between nodes - you don't need to manually persist

**State Mutation**: Always return new state dicts rather than mutating existing state in nodes

**Error Handling**: LangGraph nodes should handle exceptions gracefully to prevent workflow crashes

**See**: [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design and [README.md](README.md) for project overview
