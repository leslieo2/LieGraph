"""
Tool library for agent-based LLM workflow orchestration.

This module provides tools for:
- Identity inference and role analysis
- Speech consistency analysis
- Strategic speech generation
- Vote decision support

All tools follow the LangChain @tool pattern and use ToolRuntime
for state access with proper player-level access control.
"""

from .identity import update_player_mindset_tool
from .speech import generate_speech_tool, analyze_speech_consistency
from .voting import decide_vote_tool

__all__ = [
    "update_player_mindset_tool",
    "generate_speech_tool",
    "analyze_speech_consistency",
    "decide_vote_tool",
]
