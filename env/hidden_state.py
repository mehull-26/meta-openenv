"""True hidden-state transitions for mastery and forgetting."""

from __future__ import annotations

import math

from adaptive_learning_system.config.defaults import (
    BASE_LEARNING_BOOST,
    INCORRECT_PENALTY_FACTOR,
    INTRODUCTION_BASELINE,
)

from .types import HiddenConceptState, clamp


def mastery_to_stage(mastery: float) -> str:
    """Map a mastery estimate to the architecture's stage labels."""

    if mastery <= 0.0:
        return "unseen"
    if mastery < 0.50:
        return "learning"
    if mastery < 0.85:
        return "learned"
    return "mastered"


def apply_decay(
    hidden_state: dict[str, HiddenConceptState],
    decay_rates: dict[str, float],
    elapsed_seconds: float,
) -> None:
    """Apply exponential forgetting to every concept."""

    elapsed_days = elapsed_seconds / 86_400
    for concept_id, state in hidden_state.items():
        state.true_mastery = clamp(
            state.true_mastery * math.exp(-decay_rates[concept_id] * elapsed_days)
        )
        state.time_since_last_seconds += elapsed_seconds
        state.stage = mastery_to_stage(state.true_mastery)


def apply_learning_event(
    state: HiddenConceptState,
    outcome: str,
    action_type: str,
) -> None:
    """Update true mastery after an interaction."""

    if action_type == "Introduce":
        state.true_mastery = max(state.true_mastery, INTRODUCTION_BASELINE)

    if outcome == "correct":
        boost = BASE_LEARNING_BOOST / (1 + state.exposure_count * 0.3)
        state.true_mastery = clamp(state.true_mastery + boost)
        state.consecutive_incorrect = 0
    elif outcome == "partial":
        boost = (BASE_LEARNING_BOOST * 0.6) / (1 + state.exposure_count * 0.3)
        state.true_mastery = clamp(state.true_mastery + boost)
        state.consecutive_incorrect = 0
    elif outcome in {"incorrect", "skipped"}:
        state.true_mastery = clamp(state.true_mastery * INCORRECT_PENALTY_FACTOR)
        state.consecutive_incorrect += 1

    state.exposure_count += 1
    state.time_since_last_seconds = 0.0
    state.stage = mastery_to_stage(state.true_mastery)
