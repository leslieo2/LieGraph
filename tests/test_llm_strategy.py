import asyncio
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

from src.game.strategy import llm_update_player_mindset
from src.game.strategy.builders.prompt_builder import (
    _INFERENCE_PROMPT_PREFIX,
    format_speech_system_prompt as _format_speech_system_prompt,
)
from src.game.strategy.builders.context_builder import (
    build_inference_user_context as _build_inference_user_context,
    build_speech_user_context as _build_speech_user_context,
)

# Sample data for testing
from src.game.state import PlayerMindset, SelfBelief, Suspicion


def make_self_belief(role: str = "civilian", confidence: float = 0.5) -> SelfBelief:
    return {"role": role, "confidence": confidence}


def make_suspicion(role: str, confidence: float, reason: str) -> Suspicion:
    return {"role": role, "confidence": confidence, "reason": reason}


def make_player_mindset(
    self_belief: SelfBelief | None = None,
    suspicions: Dict[str, Suspicion] | None = None,
) -> PlayerMindset:
    return {
        "self_belief": self_belief or make_self_belief(),
        "suspicions": suspicions or {},
    }


mock_player_mindset = make_player_mindset(
    self_belief=make_self_belief(role="civilian", confidence=0.8),
    suspicions={"b": make_suspicion("spy", 0.6, "Vague speech.")},
)

# Test data for inference functions (uses playerMindset)
mock_state_inference_en = {
    "my_word": "apple",
    "completed_speeches": [
        {"player_id": "b", "content": "It's a fruit.", "round": 0, "seq": 0}
    ],
    "players": ["a", "b", "c"],
    "alive": ["a", "b", "c"],
    "me": "a",
    "rules": {"spy_count": 1},
    "existing_player_mindset": mock_player_mindset,
}

mock_state_inference_zh = {
    "my_word": "apple",
    "completed_speeches": [
        {"player_id": "b", "content": "It's a type of fruit.", "round": 0, "seq": 0}
    ],
    "players": ["a", "b", "c"],
    "alive": ["a", "b", "c"],
    "me": "a",
    "rules": {"spy_count": 1},
    "existing_player_mindset": mock_player_mindset,
}

# Test data for speech and vote functions (uses separate self_belief and suspicions)
mock_state_speech_vote_en = {
    "my_word": "apple",
    "self_belief": make_self_belief(role="civilian", confidence=0.8),
    "suspicions": {"b": make_suspicion("spy", 0.6, "Vague speech.")},
    "completed_speeches": [
        {"player_id": "b", "content": "It's a fruit.", "round": 1, "seq": 0}
    ],
    "me": "a",
    "alive": ["a", "b", "c"],
    "current_round": 1,
}

mock_state_speech_vote_zh = {
    "my_word": "apple",
    "self_belief": make_self_belief(role="civilian", confidence=0.8),
    "suspicions": {"b": make_suspicion("spy", 0.6, "Speech is very vague.")},
    "completed_speeches": [
        {"player_id": "b", "content": "It's a type of fruit.", "round": 1, "seq": 0}
    ],
    "me": "a",
    "alive": ["a", "b", "c"],
    "current_round": 1,
}

mock_speech_plan = {
    "player": "a",
    "round": 1,
    "clarity": "low",
    "clarity_reason": "LOW clarity — broad and neutral foundation",
    "goal": {
        "label": "stay_neutral",
        "reason": "Gather more evidence before committing.",
    },
    "self_role_view": "civilian",
    "self_confidence": 0.8,
    "alive_teammates": ["b", "c"],
    "top_suspicions": [
        {
            "player_id": "b",
            "suspected_role": "spy",
            "confidence": 0.6,
            "reason": "Vague speech.",
        }
    ],
}


def build_inference_prompt_for_test(
    my_word: str,
    completed_speeches: list,
    players: list,
    alive: list,
    me: str,
    rules: dict,
    existing_player_mindset: PlayerMindset,
):
    static_prompt = _INFERENCE_PROMPT_PREFIX.format(
        my_word=my_word,
        player_count=len(players),
        spy_count=rules.get("spy_count", 1),
    )
    dynamic_prompt = _build_inference_user_context(
        completed_speeches, players, alive, me, existing_player_mindset
    )
    return static_prompt + dynamic_prompt


def test_build_inference_prompt_en():
    """Tests that the English inference prompt is built correctly."""
    test_state = mock_state_inference_en.copy()
    prompt = build_inference_prompt_for_test(**test_state)
    assert "Who is the Spy" in prompt
    assert "<inference_context>" in prompt
    assert '<players me="a">' in prompt
    assert '<mindset self_role="civilian" self_confidence="0.80">' in prompt
    assert '<speech seq="0" player="b">It&#x27;s a fruit.</speech>' in prompt


def test_build_inference_prompt_zh():
    """Tests that the Chinese inference prompt is built correctly."""
    test_state = mock_state_inference_zh.copy()
    prompt = build_inference_prompt_for_test(**test_state)
    # assert "谁是卧底" in prompt  # TODO: Add translation check
    assert "Who is the Spy" in prompt
    assert '<mindset self_role="civilian" self_confidence="0.80">' in prompt
    assert '<speech seq="0" player="b">It&#x27;s a type of fruit.</speech>' in prompt


def build_speech_prompt_for_test(
    my_word: str,
    self_belief: SelfBelief,
    suspicions: dict,
    completed_speeches: list,
    me: str,
    alive: list,
    current_round: int,
    speech_plan: dict | None = None,
):
    static_prompt = _format_speech_system_prompt(my_word, self_belief)
    dynamic_prompt = _build_speech_user_context(
        self_belief, completed_speeches, me, alive, current_round, speech_plan
    )
    return static_prompt + dynamic_prompt


def test_build_speech_prompt_en():
    """Tests that the English speech prompt is built correctly."""
    test_state = mock_state_speech_vote_en.copy()
    prompt = build_speech_prompt_for_test(
        speech_plan=mock_speech_plan,
        **test_state,
    )
    assert 'Your secret word is "apple"' in prompt
    assert "<speech_context>" in prompt
    assert '<self role="civilian" confidence="0.80" />' in prompt
    assert '<strategy round="1" clarity="low">' in prompt
    assert '<planning source="plan_speech_tool">' in prompt
    assert 'goal="stay_neutral"' in prompt
    assert '<suspect id="b" role="spy" confidence="0.60">' in prompt
    assert '<speech seq="0" player="b">It&#x27;s a fruit.</speech>' in prompt


def test_build_speech_prompt_zh():
    """Tests that the Chinese speech prompt is built correctly."""
    test_state = mock_state_speech_vote_zh.copy()
    prompt = build_speech_prompt_for_test(
        speech_plan=mock_speech_plan,
        **test_state,
    )
    # assert "轮到你发言了" in prompt  # TODO: Add translation check
    assert 'Your secret word is "apple"' in prompt
    assert "<speech_context>" in prompt
    assert '<self role="civilian" confidence="0.80" />' in prompt
    assert '<strategy round="1" clarity="low">' in prompt
    assert '<planning source="plan_speech_tool">' in prompt
    assert '<speech seq="0" player="b">It&#x27;s a type of fruit.</speech>' in prompt


def test_llm_update_player_mindset_success():
    """Tests successful belief inference with structured output."""
    # Mock the agent's invoke method to return structured response
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={
            "structured_response": make_player_mindset(
                self_belief=make_self_belief("civilian", 0.9),
                suspicions={"b": make_suspicion("spy", 0.7, "Suspicious speech")},
            )
        }
    )

    # Mock create_agent to return our mock agent
    with patch("src.game.strategy.strategy_core.create_agent", return_value=mock_agent):
        mock_llm = MagicMock()
        test_state = mock_state_inference_en.copy()
        result = asyncio.run(
            llm_update_player_mindset(llm_client=mock_llm, **test_state)
        )

        assert result["self_belief"]["role"] == "civilian"
        assert result["suspicions"]["b"]["reason"] == "Suspicious speech"
        mock_agent.ainvoke.assert_awaited_once()


def test_llm_update_player_mindset_failure():
    """Tests fallback behavior when structured output extraction fails for inference."""
    # Mock the agent's invoke method to return None (simulating failure)
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"structured_response": None})

    # Mock create_agent to return our mock agent
    with patch("src.game.strategy.strategy_core.create_agent", return_value=mock_agent):
        mock_llm = MagicMock()
        test_state = mock_state_inference_en.copy()
        result = asyncio.run(
            llm_update_player_mindset(llm_client=mock_llm, **test_state)
        )

        assert result["self_belief"]["role"] == "civilian"
        assert (
            result["self_belief"]["confidence"]
            == mock_player_mindset["self_belief"]["confidence"]
        )
        mock_agent.ainvoke.assert_awaited_once()
