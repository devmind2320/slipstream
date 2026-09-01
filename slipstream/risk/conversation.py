"""Multi-turn risk accumulation (§7.3).

The brief calls this out and most implementations ignore it. Trust does not reset
per turn: a questionable output early in a conversation shapes what the model
says later, and an attacker who cannot get a bad answer in one turn will build to
it over five.

Two mechanisms:

* **Accumulator** — an exponentially decaying sum of prior turn risk. Feeds the
  estimator as one more cheap signal.
* **Ratchet** — a *floor* on the band. The band for turn N is at least the
  ratcheted floor from turns 1..N-1. The floor decays with a configurable
  half-life but never drops below the pack's ``min_band``.

Plus one hard rule: a tool call that writes or takes irreversible action forces
Band.HIGH regardless of text-level risk.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..contracts import Action, Band, BAND_ORDER, max_band
from .features import ToolScope


@dataclass
class ConversationState:
    conversation_id: str
    turn_index: int = 0
    accumulated_risk: float = 0.0
    escalations: int = 0
    agent_actions_taken: int = 0
    # Ratchet floor stored as a continuous level so it can decay smoothly
    # between band boundaries instead of stepping.
    ratchet_level: float = 0.0
    last_turn_at: float = field(default_factory=time.monotonic)
    history: list[tuple[Band, float]] = field(default_factory=list)

    def ratchet_floor(self, half_life_turns: float, min_band: Band) -> Band:
        level = self.ratchet_level
        floor = Band.LOW
        if level >= 1.5:
            floor = Band.HIGH
        elif level >= 0.5:
            floor = Band.MEDIUM
        return max_band(floor, min_band)

    def decay(self, half_life_turns: float) -> None:
        if half_life_turns <= 0:
            self.ratchet_level = 0.0
            self.accumulated_risk = 0.0
            return
        factor = math.pow(0.5, 1.0 / half_life_turns)
        self.ratchet_level *= factor
        self.accumulated_risk *= factor

    def observe(
        self,
        *,
        band: Band,
        risk: float,
        action: Action,
        tool_scope: ToolScope = ToolScope.NONE,
        half_life_turns: float = 4.0,
    ) -> None:
        """Record the outcome of a turn. Called after the response completes."""
        self.decay(half_life_turns)
        self.turn_index += 1
        self.accumulated_risk = min(1.0, self.accumulated_risk + risk)
        self.history.append((band, risk))

        # The ratchet is driven by what actually happened, not by the estimate.
        bump = float(BAND_ORDER[band]) * 0.35 + risk * 0.35
        if action in (Action.ESCALATE, Action.BLOCK):
            self.escalations += 1
            bump += 0.9
        elif action in (Action.MASK, Action.REGENERATE):
            bump += 0.5
        elif action is Action.ANNOTATE:
            bump += 0.2
        if tool_scope in (ToolScope.WRITE, ToolScope.IRREVERSIBLE):
            self.agent_actions_taken += 1
            bump += 0.8

        self.ratchet_level = min(3.0, self.ratchet_level + bump)
        self.last_turn_at = time.monotonic()


class ConversationStore:
    """In-process conversation state. Bounded so a long-running gateway does not
    grow without limit; eviction is oldest-touched-first."""

    def __init__(self, max_conversations: int = 4096) -> None:
        self._states: dict[str, ConversationState] = {}
        self.max_conversations = max_conversations

    def get(self, conversation_id: str | None) -> ConversationState:
        if not conversation_id:
            return ConversationState(conversation_id="anonymous")
        st = self._states.get(conversation_id)
        if st is None:
            if len(self._states) >= self.max_conversations:
                oldest = min(self._states.values(), key=lambda s: s.last_turn_at)
                self._states.pop(oldest.conversation_id, None)
            st = ConversationState(conversation_id=conversation_id)
            self._states[conversation_id] = st
        return st

    def reset(self, conversation_id: str | None = None) -> None:
        if conversation_id is None:
            self._states.clear()
        else:
            self._states.pop(conversation_id, None)

    def __len__(self) -> int:
        return len(self._states)


def apply_ratchet(
    turn_band: Band,
    state: ConversationState,
    *,
    half_life_turns: float,
    min_band: Band,
    tool_scope: ToolScope = ToolScope.NONE,
    tool_write_forces_high: bool = True,
) -> tuple[Band, str]:
    """Band for this turn = max(turn band, ratchet floor, pack floor).

    Returns the band and a short reason string for the decision record.
    """
    if tool_write_forces_high and tool_scope in (ToolScope.WRITE, ToolScope.IRREVERSIBLE):
        return Band.HIGH, f"tool_scope={tool_scope.value}_forces_high"

    floor = state.ratchet_floor(half_life_turns, min_band)
    band = max_band(turn_band, floor)
    if band is turn_band and floor is not turn_band:
        return band, "turn_band"
    if floor is not Band.LOW and BAND_ORDER[floor] >= BAND_ORDER[turn_band]:
        return band, f"ratchet_floor={floor.value}(level={state.ratchet_level:.2f})"
    return band, "turn_band"
