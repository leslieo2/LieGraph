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
llm_client = create_llm()


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


def get_player_context(state: GameState, player_id: str) -> Dict[str, Any]:
    private_context = state.get("player_private_states", {}).get(player_id, {})
    public_player_context = state.copy()
    public_player_context.pop("player_private_states", None)
    public_player_context.pop("host_private_state", None)

    return {"public": public_player_context, "private": private_context}


def merge_probs(old_probs: Dict[str, Any], new_probs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two probability dictionaries with incremental updates.
    """
    out = dict(old_probs)
    for pid, payload in new_probs.items():
        if hasattr(payload, "model_dump"):
            out[pid] = payload.model_dump()
        else:
            out[pid] = dict(payload)
    return out


# --- Prompt Builders ---
# Optimized for Prompt Caching: Static prefixes are defined once.

_INFERENCE_SYSTEM_PROMPT = """You are a player in the game "Who is the Spy". Your goal is to win.
- If you are a Civilian: you win by identifying and voting out the Spy.
- If you are the Spy: you win when the number of alive spies is equal to or greater than the number of alive civilians.
- Pay close attention to whether other players' descriptions match your understanding of your word.
- If descriptions seem inconsistent with your word, you might be the Spy.
- IMPORTANT: You MUST respond in the same language as the user's word, which is: "{my_word}".

**Game Rules:**
- Players: {player_count}, Spies: {spy_count}
- Civilians get one word, the Spy gets a related one.
- Describe your word each round, then vote. The player with most votes is eliminated.

**Your Task:**
Carefully review ALL previous speeches and update your PlayerMindset based on your analysis. Your `suspicions` should cover all other alive players. Treat the prior mindset as provisional evidence, not as a constraint.

**Analysis Instructions:**
1. **Review Previous Speeches**: Analyze each player's speech content for consistency with your word "{my_word}".

2. **Self-Role Analysis**:
   - **Step A: Identify Inconsistencies**: Review all speeches and categorize inconsistencies into two levels:
     - **Strong Conflicts**: Statements that are almost impossible if the speaker had your word "{my_word}". These are rare but decisive.
     - **Subtle Deviations**: Statements that are technically possible for your word, but feel slightly odd, unexpected, or point to a different "center of gravity". For example, if your word is "警察" (Police), and someone describes it as "一份需要穿制服的工作" (a job that requires a uniform), this is plausible. But if three different people all talk about "服务顾客" (serving customers) or "在店里工作" (working in a store), the collection of these subtle deviations might suggest the word is actually "店员" (clerk).

   - **Step B: Analyze the Evidence**:
     - **Evidence FOR being the Spy**:
       - List all **Strong Conflicts**. Explain why.
       - List all **Subtle Deviations**. Crucially, analyze if these deviations form a *pattern*. Do multiple players seem to be describing a slightly different object?
     - **Evidence AGAINST being the Spy**: List statements that strongly align with your word and couldn't easily apply to other related words.

   - **Step C: Conclusion**:
     - If you find even one **Strong Conflict** with a player you believe is a civilian, you MUST conclude you are the spy.
     - If you find **a recurring pattern of 2 or more Subtle Deviations** that all point to a consistent alternative concept, you should strongly consider that you might be the spy.
     - Based on the balance of evidence, decide if you are more likely "spy" or "civilian".

3. **Update Self-Belief**: Based on your conclusion:
   - **Update your role**:
       - For any strong conflict, set your own role to the opposite of the conflicting player's suspected role (per the rule above). Make this switch even if it contradicts your previous belief.
       - When conflicts are weak or ambiguous, you may remain in the previous role but justify why the evidence is inconclusive.
   - **Update confidence**: Adjust your confidence (0.0-1.0).
       - After enforcing the role flip rule, set the new role with `confidence >= 0.6`.
       - If you find a pattern of subtle deviations but no strong conflicts, decrease your `confidence` in being a "civilian" to the `0.4-0.6` range to reflect your uncertainty.
       - If evidence strongly implies you are the spy (multiple conflicts, or no one matches your word), set `confidence <= 0.4` when staying "civilian" or switch to "spy" with `confidence >= 0.6`.
       - If evidence favors you being civilian, increase confidence gradually but keep it below 0.95 unless you have overwhelming agreement.

4. **Form Suspicions**: For each alive player (except yourself), provide:
   - role: "civilian" or "spy"
   - confidence: 0.0-1.0 based on evidence
   - reason: specific reasoning based on their speech content

**Confidence Scale Guidance:**
- self_belief.confidence = 0.5: completely uncertain about your role.
- self_belief.confidence > 0.5: leaning toward the specified role.
- self_belief.confidence < 0.5: leaning toward the opposite role. Use values ≤ 0.4 to signal serious doubt when staying in the same role.
---
When you respond, use the PlayerMindset tool to return the updated state. Do not provide free-form prose outside of the tool output.
"""

# Backward compatibility for tests/imports expecting legacy constant name.
_INFERENCE_PROMPT_PREFIX = _INFERENCE_SYSTEM_PROMPT

_SPEECH_SYSTEM_PROMPT = """You are a player in the party game "Who is the Spy". It's your turn to speak.

# Language & Output
- IMPORTANT: Reply **in the same language as your word**: "{my_word}".
- Output **one line only** containing **just your speech**. No role, no reasoning, no labels, no quotes, no emojis.

# Hard Rules
1) **Do NOT lie.** Your description must be truthful and accurate for your word. For example, if your word is "牛奶" (milk), you cannot say "它是黄色的" (it is yellow), because that is a lie. The fun of the game comes from giving truthful, but skillful, descriptions.
2) **Do NOT repeat descriptions** that other players have already said in this round. Your description must be new.
3) Do NOT say the word itself or obvious synonyms that trivially reveal it.
4) Do NOT mention roles, probabilities, or game mechanics.
5) Do NOT ask questions, accuse, or name players in this speech turn (this is the description phase).
6) Stay concise.
   - If the language is Chinese/Japanese/Korean: **18–35 characters**.
   - Otherwise: **20–40 words**.

# Round Strategy Explanation
  - LOW clarity: Be broad, but include **one subtle "test" detail**. This detail should be true for your word but might be slightly awkward for a very similar word. The goal is to gently probe, not expose.
  - MEDIUM clarity: Add **1 clear differentiator**. At this stage, your description should start to actively exclude closely related words.
  - HIGH clarity: Add **1-2 strong differentiators**. Your description should make it very difficult for a spy with a related word to agree.

# Strategic Guidance
- Based on your self-belief confidence, adjust your strategy:
  - If confidence > 0.8 (highly certain you are civilian): Your job is now to **actively test** others. Your description can be a little more specific to force the spy to reveal a flaw, because they cannot lie. For example, for "海洋" (ocean), you could say "它占了地球表面的绝大部分", a fact a spy with the word "湖泊" (lake) cannot truthfully claim.
  - If confidence is between 0.5 and 0.8 (leaning civilian): Provide balanced descriptions with a "gentle differentiator".
  - If confidence < 0.5 (suspecting you are the spy): Your goal is to blend in **without lying**. Find truthful descriptions for your word that are as generic as possible. Listen to what others say and try to find the **overlapping features** between their descriptions and your own word. For example, if you think their word is "牛奶" (milk) and yours is "豆浆" (soy milk), you could truthfully say "it is a white liquid" or "it is often drunk for breakfast". Avoid details that are unique to your word and contradict the emerging consensus.
- Choose **2–3** elements: **category/purpose/scene/time/tactile feel/taste or smell/shape or color/intended user**.
- Provide **at least one** "gentle differentiator" that helps distinguish from likely near-synonyms without **pointing directly to the target**.
- Avoid brands, proper nouns, numbers, or rare specifics.
- Reuse **neutral** words and sentence patterns others have used; **mirror** the group's style to reduce mismatch.
---
Your reply must be exactly one line of plain text (no lists, no labels) in the language of "{my_word}".
"""

# Retain legacy name so existing callers/tests keep working.
_SPEECH_PROMPT_PREFIX = _SPEECH_SYSTEM_PROMPT


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
    rounds_to_keep: int = 2,
    max_entries: int = 12,
) -> str:
    if not completed_speeches:
        return "<speech_logs />"

    ordered = sorted(
        completed_speeches,
        key=lambda s: (s.get("round", 0), s.get("seq", 0)),
    )
    round_ids = sorted({speech.get("round", 0) for speech in ordered})
    selected_rounds = set(round_ids[-rounds_to_keep:])
    filtered = [
        speech for speech in ordered if speech.get("round", 0) in selected_rounds
    ]

    if len(filtered) > max_entries:
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
    my_word: str,
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    playerMindset: PlayerMindset,
) -> str:
    players_xml = _format_players_xml(players, alive, me)
    mindset_xml = _format_mindset_xml(playerMindset)
    speeches_xml = _format_speeches_xml(completed_speeches)

    return (
        "<inference_context>"
        f"<assigned_word>{escape(my_word)}</assigned_word>"
        f"{players_xml}{mindset_xml}{speeches_xml}"
        "<response_guidance>Use the PlayerMindset tool only; do not provide prose or explanations.</response_guidance>"
        "</inference_context>"
    )


def _build_inference_dynamic_suffix(
    my_word: str,
    completed_speeches: Sequence[Speech],
    players: List[str],
    alive: List[str],
    me: str,
    playerMindset: PlayerMindset,
) -> str:
    """Backward-compatible wrapper returning the structured inference context."""
    return _build_inference_user_context(
        my_word, completed_speeches, players, alive, me, playerMindset
    )


def _format_round_speeches_xml(
    completed_speeches: Sequence[Speech],
    current_round: int,
    max_entries: int = 8,
) -> str:
    round_speeches = [
        speech for speech in completed_speeches if speech.get("round") == current_round
    ]
    if not round_speeches:
        return f'<speech_round index="{current_round}" />'

    round_speeches.sort(key=lambda s: s.get("seq", 0))
    if len(round_speeches) > max_entries:
        round_speeches = round_speeches[:max_entries]

    segments = [f'<speech_round index="{current_round}">']
    for speech in round_speeches:
        segments.append(
            (
                f'<speech seq="{speech.get("seq", 0)}" '
                f'player="{escape(speech.get("player_id", "unknown"))}">'
                f"{escape(_trim_text_for_prompt(speech.get("content", ""), limit=140))}"
                "</speech>"
            )
        )

    segments.append("</speech_round>")
    return "".join(segments)


def _build_speech_user_context(
    my_word: str,
    self_belief: SelfBelief,
    suspicions: Dict[str, Suspicion],
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    """Builds the dynamic part of the speech prompt as structured XML."""
    self_role = self_belief.role
    self_confidence = self_belief.confidence

    if current_round <= 1:
        clarity_code = "low"
        clarity_desc = "LOW clarity — broad and neutral"
    elif current_round == 2:
        clarity_code = "medium"
        clarity_desc = "MEDIUM clarity — balanced specificity"
    else:
        clarity_code = "high"
        clarity_desc = "HIGH clarity — concrete but still safe"

    alive_tags = "".join(f'<player id="{escape(pid)}" />' for pid in alive)
    alive_block = f"<alive_players>{alive_tags or '<none />'}</alive_players>"

    if suspicions:
        sorted_suspicions = sorted(
            suspicions.items(), key=lambda item: item[1].confidence, reverse=True
        )
        top_id, top_suspicion = sorted_suspicions[0]
        susp_reason = _trim_text_for_prompt(top_suspicion.reason, limit=140)
        top_suspicion_block = (
            f'<top_suspicion target="{escape(top_id)}" '
            f'role="{escape(top_suspicion.role)}" '
            f'confidence="{top_suspicion.confidence:.2f}">'
            f"{escape(susp_reason)}"
            "</top_suspicion>"
        )
    else:
        top_suspicion_block = "<top_suspicion />"

    round_speeches_block = _format_round_speeches_xml(completed_speeches, current_round)

    return (
        "<speech_context>"
        f"<assigned_word>{escape(my_word)}</assigned_word>"
        f'<self role="{escape(self_role)}" confidence="{self_confidence:.2f}" />'
        f"{top_suspicion_block}"
        f'<strategy round="{current_round}" clarity="{clarity_code}">{escape(clarity_desc)}</strategy>'
        f'<speaker id="{escape(me)}" />'
        f"{alive_block}{round_speeches_block}"
        "<response_guidance>Return exactly one line of speech; avoid emojis, labels, or extra commentary.</response_guidance>"
        "</speech_context>"
    )


def _build_speech_dynamic_suffix(
    my_word: str,
    self_belief: SelfBelief,
    suspicions: Dict[str, Suspicion],
    completed_speeches: Sequence[Speech],
    me: str,
    alive: List[str],
    current_round: int,
) -> str:
    """Backward-compatible wrapper returning the structured speech context."""
    return _build_speech_user_context(
        my_word, self_belief, suspicions, completed_speeches, me, alive, current_round
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
    playerMindset: PlayerMindset,
) -> PlayerMindset:
    # Store old self_belief for logging
    old_self_belief = playerMindset.self_belief

    # 1. Format the system prompt (instructions)
    system_prompt = _INFERENCE_SYSTEM_PROMPT.format(
        my_word=my_word, player_count=len(players), spy_count=rules.get("spy_count", 1)
    )

    # 2. Build the user context (structured, dynamic state)
    user_context = _build_inference_user_context(
        my_word, completed_speeches, players, alive, me, playerMindset
    )

    extractor = create_extractor(
        llm_client, tools=[PlayerMindset], tool_choice="PlayerMindset"
    )
    result = extractor.invoke(
        {"messages": [("system", system_prompt), ("user", user_context)]}
    )

    if result["responses"]:
        new_mindset = result["responses"][0]
        log_self_belief_update(me, old_self_belief, new_mindset.self_belief)
        return new_mindset

    # Fallback: return default mindset
    default_mindset = PlayerMindset(
        self_belief=SelfBelief(role="civilian", confidence=0.5), suspicions={}
    )
    log_self_belief_update(me, old_self_belief, default_mindset.self_belief)
    return default_mindset


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
    system_prompt = _SPEECH_SYSTEM_PROMPT.format(my_word=my_word)
    user_context = _build_speech_user_context(
        my_word, self_belief, suspicions, completed_speeches, me, alive, current_round
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_context),
    ]

    response = llm_client.invoke(messages)

    raw_text = response.content if hasattr(response, "content") else response
    return _sanitize_speech_output(raw_text)
