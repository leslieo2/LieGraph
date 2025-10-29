"""Mindset inference helpers powered by LLM structured extraction."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence

from trustcall import create_extractor

from src.game.state import PlayerMindset, SelfBelief, Speech
from src.tools.llm import get_default_llm_client
from src.tools.llm._formatting import (
    format_mindset_xml,
    format_players_xml,
    format_speeches_xml,
)

__all__ = ["llm_update_player_mindset", "log_self_belief_update"]


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


def log_self_belief_update(
    player_id: str,
    old_belief: SelfBelief,
    new_belief: SelfBelief,
    timestamp: datetime | None = None,
) -> None:
    """Persist belief deltas for offline debugging."""

    if timestamp is None:
        timestamp = datetime.now()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "self_belief_updates.log")

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

    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def _build_inference_user_context(
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    existing_player_mindset: PlayerMindset,
) -> str:
    players_xml = format_players_xml(players, alive, me)
    mindset_xml = format_mindset_xml(existing_player_mindset)
    speeches_xml = format_speeches_xml(completed_speeches)

    return (
        "<inference_context>"
        f"{players_xml}{mindset_xml}{speeches_xml}"
        "<response_guidance>Use the PlayerMindset tool only; do not provide prose or explanations.</response_guidance>"
        "</inference_context>"
    )


def llm_update_player_mindset(
    *,
    llm_client: Any | None = None,
    my_word: str,
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    rules: Dict[str, Any],
    existing_player_mindset: PlayerMindset,
) -> PlayerMindset:
    existing_self_belief = existing_player_mindset.self_belief

    client = llm_client or get_default_llm_client()

    system_prompt = _INFERENCE_PROMPT_PREFIX.format(
        my_word=my_word,
        player_count=len(players),
        spy_count=rules.get("spy_count", 1),
    )

    user_context = _build_inference_user_context(
        completed_speeches, players, alive, me, existing_player_mindset
    )

    extractor = create_extractor(
        client, tools=[PlayerMindset], tool_choice="PlayerMindset"
    )
    result = extractor.invoke(
        {"messages": [("system", system_prompt), ("user", user_context)]}
    )

    if result["responses"]:
        new_mindset = result["responses"][0]
        log_self_belief_update(me, existing_self_belief, new_mindset.self_belief)
        return new_mindset

    log_self_belief_update(
        me, existing_self_belief, existing_player_mindset.self_belief
    )
    return existing_player_mindset
