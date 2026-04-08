"""Observation generation from the hidden mastery state."""

from __future__ import annotations

import random

from .types import HiddenConceptState, StudentObservation


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    threshold = rng.random() * total
    running = 0.0
    for label, weight in weights.items():
        running += weight
        if running >= threshold:
            return label
    return next(iter(weights))


def generate_observation(
    rng: random.Random,
    concept_id: str,
    state: HiddenConceptState,
    action_type: str,
) -> StudentObservation:
    """Sample a noisy student interaction from hidden mastery."""

    mastery = state.true_mastery
    if action_type == "Introduce":
        mastery = min(1.0, mastery + 0.15)

    outcome_weights = {
        "correct": max(0.05, mastery),
        "incorrect": max(0.05, 1.0 - mastery),
        "partial": max(0.05, 1.0 - abs(mastery - 0.5) * 2.0),
        "skipped": max(0.02, (1.0 - mastery) ** 2),
    }
    if action_type == "Review":
        outcome_weights["correct"] += 0.08
    if action_type == "Assess":
        outcome_weights["incorrect"] += 0.05

    outcome = _weighted_choice(rng, outcome_weights)
    response_time_seconds = max(
        4.0,
        7.5 + (1.0 - mastery) * 18.0 + rng.uniform(-2.0, 2.0),
    )
    hint_used = rng.random() < max(0.05, 0.55 - mastery)

    return StudentObservation(
        concept_id=concept_id,
        outcome=outcome,  # type: ignore[arg-type]
        response_time_seconds=response_time_seconds,
        hint_used=hint_used,
        gap_seconds=state.time_since_last_seconds,
    )
