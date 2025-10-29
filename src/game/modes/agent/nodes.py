"""Agent mode node entry points."""

from __future__ import annotations

from ..registry import get_host_behavior, get_player_behavior
from ..shared.interfaces import (
    BehaviorResult,
    HostNodeContext,
    PlayerNodeContext,
)

_MODE = "agent"


def host_setup(ctx: HostNodeContext) -> BehaviorResult:
    """Execute host setup for agent mode."""

    behavior = get_host_behavior(mode=_MODE)
    return behavior.setup(ctx)


def host_stage_switch(ctx: HostNodeContext) -> BehaviorResult:
    """Advance host stage for agent mode."""

    behavior = get_host_behavior(mode=_MODE)
    return behavior.stage_switch(ctx)


def host_result(ctx: HostNodeContext) -> BehaviorResult:
    """Resolve host round results for agent mode."""

    behavior = get_host_behavior(mode=_MODE)
    return behavior.resolve_round(ctx)


def player_speech(ctx: PlayerNodeContext) -> BehaviorResult:
    """Execute player speech for agent mode."""

    behavior = get_player_behavior(ctx.player_id, mode=_MODE)
    return behavior.decide_speech(ctx)


def player_vote(ctx: PlayerNodeContext) -> BehaviorResult:
    """Execute player vote for agent mode."""

    behavior = get_player_behavior(ctx.player_id, mode=_MODE)
    return behavior.decide_vote(ctx)
