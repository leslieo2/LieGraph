"""
Identity inference and role analysis tools.

This module provides tools for players to analyze game state
and update their beliefs about their own role and other players' roles.

Access Control:
- Players can only access their own private state
- Players can access public game state (speeches, votes, etc.)
- Players cannot access other players' private states or host private state
"""

from typing import Any, Dict, List, Sequence

from src.game.state import GameState, PlayerMindset, Speech, SelfBelief, Suspicion
from trustcall import create_extractor


def _format_players_xml(players: Sequence[str], alive: Sequence[str], me: str) -> str:
    """Format player list as XML for prompt."""
    from html import escape

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


def _trim_text_for_prompt(text: str, limit: int = 180) -> str:
    """Normalize whitespace and trim long passages for prompt friendliness."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _format_mindset_xml(playerMindset: PlayerMindset) -> str:
    """Format player mindset as XML for prompt."""
    from html import escape

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
    from html import escape

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
    """Build user context for inference prompt."""
    players_xml = _format_players_xml(players, alive, me)
    mindset_xml = _format_mindset_xml(existing_player_mindset)
    speeches_xml = _format_speeches_xml(completed_speeches)

    return (
        "<inference_context>"
        f"{players_xml}{mindset_xml}{speeches_xml}"
        "<response_guidance>Use the PlayerMindset tool only; do not provide prose or explanations.</response_guidance>"
        "</inference_context>"
    )


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


def update_player_mindset_tool(
    state: GameState,
    player_id: str,
) -> Dict[str, Any]:
    """
    Analyze game state and update player's beliefs about roles.

    This tool performs identity inference by analyzing:
    - Speech consistency across players
    - Alignment with player's assigned word
    - Detection of outliers and consensus

    Access Control:
    - Can only access player's own private state (player_private_states[player_id])
    - Can access public game state (completed_speeches, players, etc.)
    - Cannot access other players' private states or host_private_state

    Args:
        state: Current GameState
        player_id: ID of the player performing the analysis

    Returns:
        Dictionary with updated player_private_states
    """

    # Access control: validate player_id exists
    if player_id not in state.get("player_private_states", {}):
        raise ValueError(f"Invalid player_id: {player_id}")

    # Get player's private state (allowed)
    player_private = state["player_private_states"][player_id]
    my_word = player_private.assigned_word
    existing_mindset = player_private.playerMindset

    # Get public game state (allowed)
    completed_speeches = state.get("completed_speeches", [])
    players = state["players"]

    # Get alive players from public state
    eliminated = set(state.get("eliminated_players", []))
    alive = [p for p in players if p not in eliminated]

    # Get game rules from config
    from src.game.config import get_config

    config = get_config()
    rules = config.get_game_rules()

    # Build prompt context
    system_prompt = _INFERENCE_PROMPT_PREFIX.format(
        my_word=my_word, player_count=len(players), spy_count=rules.get("spy_count", 1)
    )

    user_context = _build_inference_user_context(
        completed_speeches, players, alive, player_id, existing_mindset
    )

    # Call LLM with structured output extraction
    from src.tools.llm import create_llm

    llm_client = create_llm()

    extractor = create_extractor(
        llm_client, tools=[PlayerMindset], tool_choice="PlayerMindset"
    )
    result = extractor.invoke(
        {"messages": [("system", system_prompt), ("user", user_context)]}
    )

    if result["responses"]:
        new_mindset = result["responses"][0]

        # Log the update
        from src.game.llm_strategy import log_self_belief_update

        log_self_belief_update(
            player_id, existing_mindset.self_belief, new_mindset.self_belief
        )

        # Merge suspicions (preserve existing + add new)
        from src.game.state import merge_probs, PlayerPrivateState

        merged_suspicions = merge_probs(
            existing_mindset.suspicions, new_mindset.suspicions
        )

        updated_private_state = PlayerPrivateState(
            assigned_word=my_word,
            playerMindset=PlayerMindset(
                self_belief=new_mindset.self_belief,
                suspicions=merged_suspicions,
            ),
        )

        # Return update dictionary
        return {"player_private_states": {player_id: updated_private_state}}
    else:
        # LLM failed, preserve existing mindset
        from src.game.llm_strategy import log_self_belief_update

        log_self_belief_update(
            player_id, existing_mindset.self_belief, existing_mindset.self_belief
        )

        return {}


def analyze_speech_consistency(
    state: GameState,
    player_id: str,
    target_player_id: str,
) -> Dict[str, Any]:
    """
    Analyze speech consistency of a target player.

    This tool analyzes:
    - Whether target's speeches align with group consensus
    - Whether target is an outlier
    - Speech vagueness and specificity

    Access Control:
    - Can only access public speeches
    - Cannot access any private states

    Args:
        state: Current GameState
        player_id: ID of the player performing analysis
        target_player_id: ID of the player to analyze

    Returns:
        Analysis results with consistency scores and patterns
    """

    # Access control: validate both player IDs
    if player_id not in state.get("player_private_states", {}):
        raise ValueError(f"Invalid player_id: {player_id}")
    if target_player_id not in state["players"]:
        raise ValueError(f"Invalid target_player_id: {target_player_id}")

    # Get target player's speeches (public data)
    completed_speeches = state.get("completed_speeches", [])
    target_speeches = [
        s for s in completed_speeches if s.get("player_id") == target_player_id
    ]

    # Simple consistency analysis
    # TODO: This is a placeholder - could be enhanced with LLM-based analysis
    avg_length = sum(len(s.get("content", "")) for s in target_speeches) / max(
        len(target_speeches), 1
    )

    analysis = {
        "target_player_id": target_player_id,
        "speech_count": len(target_speeches),
        "speeches": [s.get("content", "") for s in target_speeches],
        "target_speeches": [
            s.get("content", "") for s in target_speeches
        ],  # Alias for consistency
        "avg_speech_length": avg_length,
        "is_outlier": False,  # Placeholder logic
        "vagueness_score": 0.5,  # Placeholder score
    }

    return analysis
