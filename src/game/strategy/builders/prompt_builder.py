"""
Prompt engineering for LLM-powered game strategies.

Manages all prompt templates and role-specific strategy determination.
"""

from typing import Dict

from src.game.state import SelfBelief
from src.game.strategy.serialization import to_plain_dict


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
- Review the <planning> tag generated via the `plan_speech` tool and align your clue with its goal.
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
- Follow the <strategy> tag in the <speech_context> tag and mirror the group's clarity while masking differences.
- Review the <planning> tag generated via the `plan_speech` tool and align your clue with its goal.
- If you sense conflict with the group, emphasize broad categories, shared settings, or emotions instead of specifics.
- Choose 2-3 aspects such as category, purpose, setting, sensory detail, or user that civilians might also mention.
- Mirror the tone and vocabulary other players use.
- Avoid brands, numbers, and rare trivia unless essential.
Reply now with your single-line speech."""

_VOTE_PROMPT_PREFIX = """You are playing "Who is the Spy" and it is time to vote.
Your secret word is "{my_word}".
Decide between two voting strategies, and call exactly one tool:
- `decide_player_vote`: Use when one player feels clearly more suspicious or the game is already in later rounds.
- `decide_player_vote_second_best`: Use when suspicions are close together, you are still in the first two rounds, or you want to stay less predictable.
Do not call both tools. Make your internal choice, invoke the tool, then return only the player ID via the VoteDecision structured response.
(Alive players: {alive_count}, current round: {current_round})"""


def determine_clarity(
    role: str, self_confidence: float, current_round: int
) -> tuple[str, str]:
    """Return role-aware clarity code and description for the current round."""
    # TODO: When plan_speech fully controls clarity selection, collapse this helper
    # into the planning workflow to avoid maintaining duplicate heuristics.
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


def format_speech_system_prompt(my_word: str, self_belief: SelfBelief) -> str:
    """Select the civilian or spy speech prompt based on calibrated confidence."""
    belief_dict = to_plain_dict(
        self_belief,
        lambda: {"role": "civilian", "confidence": 0.0},
    )

    is_confident_spy = (
        belief_dict.get("role") == "spy"
        and float(belief_dict.get("confidence", 0.0)) >= 0.7
    )
    if is_confident_spy:
        template = _SPY_SPEECH_PROMPT_PREFIX
    else:
        template = _CIVILIAN_SPEECH_PROMPT_PREFIX
    return template.format(my_word=my_word)


def format_inference_system_prompt(
    my_word: str, player_count: int, spy_count: int
) -> str:
    """Format the inference system prompt with game parameters."""
    return _INFERENCE_PROMPT_PREFIX.format(
        my_word=my_word, player_count=player_count, spy_count=spy_count
    )


def format_vote_system_prompt(
    my_word: str, alive_count: int, current_round: int
) -> str:
    """Format system prompt for voting decisions."""
    return _VOTE_PROMPT_PREFIX.format(
        my_word=my_word, alive_count=alive_count, current_round=current_round
    )
