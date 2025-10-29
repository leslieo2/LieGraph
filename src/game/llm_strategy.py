"""
LLM-powered strategy and reasoning for the "Who Is Spy" game.

This module implements the AI agent intelligence system that enables players to:
- Dynamically infer their own identity through conversation analysis
- Generate strategic speech based on confidence levels
- Form suspicions about other players with confidence scores
- Make voting decisions based on accumulated evidence

Key Features:
- Dynamic Identity Inference: Real-time role analysis through speech patterns
- Probabilistic Belief System: Self-belief and suspicions with confidence scores
- Strategic Speech Generation: Adaptive descriptions based on game phase
- Multi-language Support: Automatic language detection and response generation

Architecture:
- Prompt Engineering: Static prefixes with dynamic context injection
- Chain of Thought: Structured reasoning for identity inference
- Confidence Calibration: Systematic confidence adjustment based on evidence
- Logging: Debug logging for belief evolution tracking
"""

import json
import os
import re
from datetime import datetime
from html import escape
from typing import List, Dict, Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from trustcall import create_extractor

from src.game.state import GameState, Speech, PlayerMindset, Suspicion, SelfBelief
from src.tools.llm import create_llm

# Game rules are now managed by the configuration system
# Use config.get_game_rules() to get game rules

# --- LLM Clients ---
# Use lazy loading to avoid initialization issues during testing
_llm_client = None


def _get_llm_client():
    """Get or create the LLM client with lazy initialization."""
    global _llm_client
    if _llm_client is None:
        _llm_client = create_llm()
    return _llm_client


# --- Helper Functions ---


def log_self_belief_update(
    player_id: str,
    old_belief: SelfBelief,
    new_belief: SelfBelief,
    timestamp: datetime = None,
):
    """
    Log self_belief updates to a file for debugging.

    Args:
        player_id: ID of the player
        old_belief: Previous self_belief state
        new_belief: Updated self_belief state
        timestamp: Optional timestamp for the update
    """
    if timestamp is None:
        timestamp = datetime.now()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"self_belief_updates.log")

    log_entry = {
        "timestamp": timestamp.isoformat(),
        "player_id": player_id,
        "old_belief": {"role": old_belief.role, "confidence": old_belief.confidence},
        "new_belief": {"role": new_belief.role, "confidence": new_belief.confidence},
        "change": {
            "role_changed": old_belief.role != new_belief.role,
            "confidence_delta": new_belief.confidence - old_belief.confidence,
        },
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# --- Prompt Builders ---
# Optimized for Prompt Caching: Static prefixes are defined once.


def _determine_clarity(
    role: str, self_confidence, current_round: int
) -> tuple[str, str]:
    """Return role-aware clarity code and description for the current round."""
    if role == "spy" and self_confidence > 0.5:
        if current_round <= 2:
            return (
                "low",
                "LOW clarity — stay broad to blend with civilians",
            )
        if current_round <= 4:
            return (
                "medium",
                "MEDIUM clarity — add safe overlaps without exposing differences",
            )
        return (
            "medium",
            "MEDIUM clarity — stay measured while matching the group's detail level",
        )

    # Civilian defaults
    if current_round <= 1:
        return "low", "LOW clarity — broad and neutral foundation"
    if current_round == 2:
        return "medium", "MEDIUM clarity — start introducing gentle differentiators"
    return "high", "HIGH clarity — press with confident, specific traits"


_INFERENCE_PROMPT_PREFIX = """You are a player in the game "Who is the Spy". Your goal is to analyze the game state and update your beliefs.
- Pay close attention to whether other players' descriptions match your understanding of your word.
- If descriptions seem inconsistent with your word, you might be the Spy.
- IMPORTANT: You MUST respond in the same language as the user's word, which is: "{my_word}".

**Game Rules:**
- Players: {player_count}, Spies: {spy_count}
- Civilians get one word, the Spy gets a related one.
- Your word is: "{my_word}"
- You MUST respond in the same language as your word.

**Your Task:**
1.  First, complete the `<thinking>` block to structure your analysis.
2.  Then, based on your analysis, use the `PlayerMindset` tool to output your updated beliefs.

---
<thinking>
**1. Self-Role Analysis:**  
*   **Evidence FOR being SPY:**  
    *   **Group Consensus Conflict:** (Do multiple players' statements align with each other but conflict with your word "{my_word}"? List them.)    
    *   **Other Inconsistencies:** (List any other statements that feel odd or point to a different concept.)
*   **Evidence FOR being CIVILIAN:**  
    *   **Outlier Conflict:** (Did one player make a statement that conflicts with both your word AND other players' statements? Identify this outlier.)    
    *   **Strong Alignment:** (List statements from others that perfectly match your word( "{my_word}".)
    *   **Conclusion:** (Based on the evidence above, are you more likely the Spy or a Civilian? State your conclusion in one sentence.)  
**2. Suspicion Analysis (for each other alive player):**  
*   **Player [ID]:**  
    *   **Evidence:** (Analyze their speech. Is it consistent with the group? Is it an outlier? Is it vague?)    
    *   **Conclusion:** (Based on the evidence, are they likely a Spy or a Civilian?)
    *   **Player [ID]:**  
    *   ... (Repeat for all other alive players)
</thinking>
---

**Decision & Confidence Rules:**
- **Self-Role:**
    - Treat strong conflicts (two or more players matching each other while clashing with your word) as a **major clue** that you might be the Spy.
    - If the evidence is mixed, stay uncertain and keep probing; do **not** force a Spy conclusion if the group could still share your concept.
- **Self-Confidence:**
    - If you are convinced you are the **Spy**, set confidence to **0.8** (very certain but still cautious).
    - If you lean Civilian and have a clear suspect, set confidence to **0.75**.
    - If the evidence is ambiguous, keep confidence around **0.5–0.65** to reflect doubt.
- **Suspicions:**
    - Strong outliers (conflicting with both you and the group) are prime Spy candidates. Mark them as **Spy** with confidence **0.85**.
    - Very vague speakers earn light suspicion. Mark them as **Spy** around **0.55**.
    - Players aligned with the consensus should be tagged **Civilian** with confidence **0.75**.

**Final Instruction:**
Now, use the `PlayerMindset` tool to return the updated state based on your completed analysis. Do not provide any other text outside the tool output.
"""

_CIVILIAN_SPEECH_PROMPT_PREFIX = """You are a civilian player in the party game "Who is the Spy". Your secret word is "{my_word}" and it is your turn to speak.
Goal: Share one truthful clue that helps fellow civilians test the group without giving away the exact word.
Must:
- Reply in the same language as "{my_word}".
- Output exactly one line of plain text; no labels, emojis, quotes, or meta reasoning.
- Tell the truth about your word; do not say the word itself or obvious synonyms.
- Do not mention roles, probabilities, mechanics, questions, accusations, or player names.
- Avoid repeating another player's description this round.
- Stay concise: 18-35 characters for Chinese/Japanese/Korean, otherwise 20-40 words.
Guide:
- Follow the <strategy> tag in the <speech_context> tag to match the desired clarity for this turn.
- Use the confidence value in the <self> tag to decide how bold to be: higher confidence supports sharper differentiators, lower confidence favors safer overlaps.
- Choose 2-3 aspects such as category, purpose, setting, sensory detail, or user.
- Mirror the tone and vocabulary other players use.
- Skip brands, numbers, and rare trivia unless essential.
Reply now with your single-line speech."""

_SPY_SPEECH_PROMPT_PREFIX = """You are the spy in the party game "Who is the Spy". Your secret word is "{my_word}" and it is your turn to speak.
Goal: Blend in by giving a plausible clue that could also fit the civilians' word while safely disguising your own.
Must:
- Reply in the same language as "{my_word}".
- Output exactly one line of plain text; no labels, emojis, quotes, or meta reasoning.
- Prioritize overlap with likely civilian clues; you may soften or generalize details to avoid exposing your unique angle.
- Do not mention roles, probabilities, mechanics, questions, accusations, or player names.
- Avoid repeating another player's description this round.
- Stay concise: 18-35 characters for Chinese/Japanese/Korean, otherwise 20-40 words.
Guide:
- Follow the <strategy> tag in the <speech_context> tag and mirror the group’s clarity while masking differences.
- If you sense conflict with the group, emphasize broad categories, shared settings, or emotions instead of specifics.
- Choose 2-3 aspects such as category, purpose, setting, sensory detail, or user that civilians might also mention.
- Mirror the tone and vocabulary other players use.
- Avoid brands, numbers, and rare trivia unless essential.
Reply now with your single-line speech."""


def _trim_text_for_prompt(text: str, limit: int = 180) -> str:
    """Normalize whitespace and trim long passages for prompt friendliness."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _format_players_xml(players: Sequence[str], alive: Sequence[str], me: str) -> str:
    alive_tags = "".join(
        f'<player id="{escape(pid)}" status="alive" />' for pid in alive
    )
    roster_tags = "".join(f'<player id="{escape(pid)}" />' for pid in players)
    return (
        f'<players me="{escape(me)}">'
        f"<alive>{alive_tags or '<none />'}</alive>"
        f"<all>{roster_tags}</all>"
        "</players>"
    )


def _format_mindset_xml(playerMindset: PlayerMindset) -> str:
    self_belief = playerMindset.self_belief
    suspicions = playerMindset.suspicions or {}
    suspicions_tags = []
    for pid, suspicion in suspicions.items():
        trimmed_reason = _trim_text_for_prompt(suspicion.reason, limit=160)
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


def _format_speeches_xml(
    completed_speeches: Sequence[Speech],
    rounds_to_keep: int | None = None,
    max_entries: int | None = None,
) -> str:
    """Render speeches in chronological order with optional trimming."""
    if not completed_speeches:
        return "<speech_logs />"

    ordered = sorted(
        completed_speeches,
        key=lambda s: (s.get("round", 0), s.get("seq", 0)),
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

    segments = ["<speech_logs>"]
    current_round = None

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
                f"{escape(_trim_text_for_prompt(speech.get("content", ""), limit=140))}"
                "</speech>"
            )
        )

    if current_round is not None:
        segments.append("</round>")

    segments.append("</speech_logs>")
    return "".join(segments)


def _build_inference_user_context(
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    existing_player_mindset: PlayerMindset,
) -> str:
    players_xml = _format_players_xml(players, alive, me)
    mindset_xml = _format_mindset_xml(existing_player_mindset)
    speeches_xml = _format_speeches_xml(completed_speeches)

    return (
        "<inference_context>"
        f"{players_xml}{mindset_xml}{speeches_xml}"
        "<response_guidance>Use the PlayerMindset tool only; do not provide prose or explanations.</response_guidance>"
        "</inference_context>"
    )


def _build_speech_user_context(
    self_belief: SelfBelief,
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    """Builds the dynamic part of the speech prompt as structured XML."""
    self_role = self_belief.role
    self_confidence = self_belief.confidence

    clarity_code, clarity_desc = _determine_clarity(
        self_role, self_confidence, current_round
    )

    alive_tags = "".join(f'<player id="{escape(pid)}" />' for pid in alive)
    alive_block = f"<alive_players>{alive_tags or '<none />'}</alive_players>"

    speech_logs_block = _format_speeches_xml(completed_speeches)

    return (
        "<speech_context>"
        f'<self role="{escape(self_role)}" confidence="{self_confidence:.2f}" />'
        f'<strategy round="{current_round}" clarity="{clarity_code}">{escape(clarity_desc)}</strategy>'
        f'<speaker id="{escape(me)}" />'
        f'{alive_block}<current_round index="{current_round}" />{speech_logs_block}'
        "<response_guidance>Return exactly one line of speech; avoid emojis, labels, or extra commentary.</response_guidance>"
        "</speech_context>"
    )


_EMOJI_REGEX = re.compile("[\u2600-\u26ff\u2700-\u27bf\U0001f300-\U0001faff]")


def _sanitize_speech_output(text: Any) -> str:
    """Flatten whitespace, drop emojis, and enforce a single-line speech."""
    if text is None:
        return ""

    raw = str(text)
    lines = [
        line.strip() for line in raw.replace("\r", "").splitlines() if line.strip()
    ]
    candidate = lines[-1] if lines else raw.strip()
    candidate = candidate.replace("\n", " ")
    candidate = _EMOJI_REGEX.sub("", candidate)
    candidate = " ".join(candidate.split())
    return candidate


# --- LLM Interaction Functions ---


def llm_update_player_mindset(
    llm_client: Any,
    my_word: str,
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    rules: Dict[str, Any],
    existing_player_mindset: PlayerMindset,
) -> PlayerMindset:
    existing_self_belief = existing_player_mindset.self_belief

    # 1. Format the system prompt (instructions)
    system_prompt = _INFERENCE_PROMPT_PREFIX.format(
        my_word=my_word, player_count=len(players), spy_count=rules.get("spy_count", 1)
    )

    # 2. Build the user context (structured, dynamic state)
    user_context = _build_inference_user_context(
        completed_speeches, players, alive, me, existing_player_mindset
    )

    extractor = create_extractor(
        llm_client, tools=[PlayerMindset], tool_choice="PlayerMindset"
    )
    result = extractor.invoke(
        {"messages": [("system", system_prompt), ("user", user_context)]}
    )

    if result["responses"]:
        new_mindset = result["responses"][0]
        log_self_belief_update(me, existing_self_belief, new_mindset.self_belief)
        return new_mindset

    # Fallback: LLM failed, preserve previous mindset
    log_self_belief_update(
        me, existing_self_belief, existing_player_mindset.self_belief
    )
    return existing_player_mindset


def llm_generate_speech(
    llm_client: Any,
    my_word: str,
    self_belief: SelfBelief,
    suspicions: Dict[str, Suspicion],
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    system_prompt = _format_speech_system_prompt(my_word, self_belief)
    user_context = _build_speech_user_context(
        self_belief, completed_speeches, me, alive, current_round
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_context),
    ]

    client = llm_client if llm_client is not None else _get_llm_client()
    response = client.invoke(messages)

    raw_text = response.content if hasattr(response, "content") else response
    return _sanitize_speech_output(raw_text)


def _format_speech_system_prompt(my_word: str, self_belief: SelfBelief) -> str:
    """Select the civilian or spy speech prompt based on calibrated confidence."""
    is_confident_spy = self_belief.role == "spy" and self_belief.confidence >= 0.7
    if is_confident_spy:
        template = _SPY_SPEECH_PROMPT_PREFIX
    else:
        template = _CIVILIAN_SPEECH_PROMPT_PREFIX
    return template.format(my_word=my_word)
