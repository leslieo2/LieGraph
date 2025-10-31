"""
Pydantic models used for structured LLM interactions.

These models are intentionally separated from the shared game state so that
LangGraph nodes can operate on plain dictionaries while the LLM integration
retains strict validation and type guarantees.
"""

from typing import Dict, Literal

from pydantic import BaseModel, Field


class SelfBeliefModel(BaseModel):
    """Structured representation of a player's belief about their own role."""

    role: Literal["civilian", "spy"]
    confidence: float = Field(ge=0.0, le=1.0)


class SuspicionModel(BaseModel):
    """Structured representation of a player's suspicion about another player."""

    role: Literal["civilian", "spy"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class PlayerMindsetModel(BaseModel):
    """Structured representation of a player's mindset for LLM responses."""

    self_belief: SelfBeliefModel
    suspicions: Dict[str, SuspicionModel] = Field(default_factory=dict)


class VoteDecisionModel(BaseModel):
    """Structured output model capturing a player's vote target."""

    target: str = Field(..., description="ID of the player to vote for.")
