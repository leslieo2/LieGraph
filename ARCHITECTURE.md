# LieGraph Architecture Documentation

## Overview

LieGraph is an AI-powered implementation of the social deduction game "Who Is Spy" built with LangGraph. It features autonomous AI agents that can reason, strategize, and interact in natural language to find the spy among them.

## High-Level Architecture

### System Design Philosophy

LieGraph follows a **state machine architecture** built on LangGraph, with clear separation between:
- **Workflow Orchestration**: Game flow management through state transitions
- **State Management**: Immutable state updates with conflict resolution
- **AI Intelligence**: LLM-powered reasoning and strategy
- **Game Logic**: Core game mechanics and rules

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI Layer (React)                     │
├─────────────────────────────────────────────────────────────┤
│                LangGraph Workflow Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Setup     │  │  Speaking   │  │      Voting         │  │
│  │   Nodes     │  │   Nodes     │  │       Nodes         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 State Management Layer                      │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │   Shared State  │  │        Private States            │  │
│  │  (GameState)    │  │  (Host + Player Mindsets)        │  │
│  └─────────────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  AI Strategy Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  Identity       │  │  Speech         │  │   Voting    │  │
│  │  Inference      │  │  Generation     │  │   Logic     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Configuration Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Game Rules    │  │   Vocabulary    │  │   Players   │  │
│  │   & Settings    │  │   Management    │  │   & Names   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. LangGraph Workflow (`src/game/graph.py`)

**Purpose**: Orchestrates the complete game flow as a state machine.

**Key Features**:
- **StateGraph Container**: Main workflow with typed state management
- **Conditional Routing**: Dynamic phase transitions based on game state
- **Concurrent Execution**: Parallel voting with sequential speaking phases
- **Node Registration**: Dynamic node creation for each player

**Architecture Patterns**:
- **Fan-out/Fan-in**: Concurrent voting nodes converge to single transition node
- **Conditional Edges**: Phase-based routing with fallback mechanisms
- **State Persistence**: Checkpointing for long-running games

### 2. State Management (`src/game/state.py`)

**Purpose**: Defines and manages the game's shared and private state.

**Data Structures**:
- **GameState TypedDict**: Central shared state with typed fields
- **Private States**: Host (invariant setup) and Player (mental models) states
- **Immutable Actions**: Speech and Vote records with timestamps

**Reducer System**:
- `merge_votes`: Timestamp-based conflict resolution
- `merge_private_states`: Incremental mindset updates
- `add`: Append-only operations for immutable records

### 3. AI Strategy (`src/tools/llm/`)

**Purpose**: Houses reusable LLM clients plus inference and speech strategies shared across modes.

**Core Intelligence Systems**:
- **Dynamic Identity Inference**: Real-time role analysis through conversation patterns
- **Probabilistic Belief System**: Confidence-based self-belief and suspicions
- **Strategic Adaptation**: Round-based and confidence-based strategy selection
- **Multi-language Reasoning**: Automatic language detection and response generation

**Prompt Engineering**:
- **Static Prefixes**: Unchanging game rules and strategy guidelines
- **Dynamic Context**: Real-time game state and player mindset injection
- **Chain of Thought**: Structured reasoning for identity inference

### 4. Game Logic (`src/game/rules.py`)

**Purpose**: Implements core game mechanics and win conditions.

**Key Functions**:
- **Role Assignment**: Random spy/civilian assignment with word distribution
- **Vote Processing**: Majority voting with tie-breaking logic
- **Win Detection**: Civilian and spy victory condition checking

### 5. Player Nodes (`src/game/nodes/host.py` & `src/game/nodes/player.py`)

**Purpose**: Thin LangGraph adapters that resolve the active mode and delegate to mode-specific implementations.

**Responsibilities**:
- Use `resolve_behavior_mode` to derive the active mode from the running state or configuration defaults.
- Construct shared `HostNodeContext` and `PlayerNodeContext` instances with LLM helper extras.
- Forward execution to `src.game.modes.<mode>.nodes`, which in turn invoke the registered behaviors.

### 6. Mode Packages (`src/game/modes/`)

**Purpose**: Encapsulate all behavior, toolbox, and node logic for each execution mode.

**Layout**:
- `shared/`: Behavior protocols, node contexts, and the `BehaviorRegistry`.
- `workflow/`: Deterministic workflow behaviors plus node wrappers that call into the registry.
- `agent/`: Agent behaviors with memory, strategy tooling, and reusable toolboxes.

**Registry Pattern**:
- `modes/registry.py` hosts `create_behavior_registry`, `get_*_behavior`, and `register_behavior_registry`.
- Modes resolve behaviors through the registry so overrides apply consistently across nodes and tooling.

### 7. Configuration (`src/game/config.py`)

**Purpose**: Centralized configuration management with validation.

**Configuration Areas**:
- **Game Settings**: Player counts, round limits, balancing rules
- **Vocabulary**: Word pairs for civilian/spy assignments
- **Player Management**: Name pools and selection logic

## Data Flow Architecture

### Game Flow Sequence

```
Initial State
    ↓
Host Setup (Role & Word Assignment)
    ↓
Speaking Phase (Sequential)
    ├── Player 1: Speech + Mindset Update
    ├── Player 2: Speech + Mindset Update
    └── ...
    ↓
Voting Phase (Concurrent)
    ├── Player 1: Vote Decision
    ├── Player 2: Vote Decision
    └── ...
    ↓
Vote Processing & Elimination
    ↓
Win Condition Check
    ↓
Next Round or Game End
```

### State Transition Patterns

**Speaking Phase**:
- Sequential execution with state accumulation
- Append-only speech records
- Incremental mindset updates

**Voting Phase**:
- Concurrent execution with reducer-based merging
- Timestamp-based vote conflict resolution
- Phase-specific vote validation

## Integration Architecture

### LangGraph Integration
- **StateGraph Foundation**: Built on LangGraph's state machine capabilities
- **Conditional Routing**: Dynamic edge selection based on game phase
- **Concurrent Execution**: Proper reducer design for parallel node execution
- **Checkpoint System**: State persistence and recovery

### LLM Integration
- **TrustCall Framework**: Structured output extraction from LLM responses
- **Prompt Caching**: Static prefix optimization for performance
- **Error Handling**: Fallback mechanisms for LLM failures
- **Multi-provider Support**: Configurable LLM client factory

### Web UI Integration
- **Real-time State**: Live game state visualization
- **Multi-language Interface**: English/Chinese support
- **Player Status Tracking**: Alive/eliminated state management

## System Properties

### Scalability
- **State Size Management**: Minimal shared state with private state separation
- **Concurrent Safety**: Reducer-based conflict resolution
- **Memory Efficiency**: Checkpointing for long-running sessions

### Extensibility
- **Modular Design**: Clear separation between workflow, state, and AI layers
- **Configuration-driven**: Game rules and settings externalized
- **Plugin Architecture**: Easy addition of new game phases and AI strategies

### Reliability
- **Immutable State**: Append-only records for critical game actions
- **Error Recovery**: Checkpoint-based state recovery
- **Validation**: Configuration and state validation at runtime
