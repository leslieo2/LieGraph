"""
Logging utilities for tracking game state changes.

Provides structured logging for debugging AI agent belief evolution.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from src.game.state import SelfBelief
from src.game.strategy.serialization import to_plain_dict


def _belief_to_dict(belief: SelfBelief) -> Dict[str, Any]:
    """Convert belief data into a plain dictionary."""
    return to_plain_dict(
        belief,
        lambda: {"role": "civilian", "confidence": 0.0},
    )


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

    old_data = _belief_to_dict(old_belief)
    new_data = _belief_to_dict(new_belief)

    old_role = old_data.get("role")
    new_role = new_data.get("role")
    old_conf = float(old_data.get("confidence", 0.0))
    new_conf = float(new_data.get("confidence", 0.0))

    log_entry = {
        "timestamp": timestamp.isoformat(),
        "player_id": player_id,
        "old_belief": {"role": old_role, "confidence": old_conf},
        "new_belief": {"role": new_role, "confidence": new_conf},
        "change": {
            "role_changed": old_role != new_role,
            "confidence_delta": new_conf - old_conf,
        },
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
