"""Belief prediction and Bayesian-style updates."""

from __future__ import annotations

import math

from adaptive_learning_system.config.defaults import (
    ADAPTATION_ALPHA,
    ASSESSMENT_READY_THRESHOLD,
    BASE_ADAPTATION_RATE,
)

from .types import BeliefConceptState, StudentObservation, clamp

GRID = [index / 20 for index in range(21)]


def apply_decay(
    beliefs: dict[str, BeliefConceptState],
    elapsed_seconds: float,
) -> None:
    """Predict the belief prior after time passes."""

    elapsed_days = elapsed_seconds / 86_400
    for belief in beliefs.values():
        belief.estimated_mastery = clamp(
            belief.estimated_mastery * math.exp(-belief.decay_rate * elapsed_days)
        )


def _observation_likelihood(point: float, observation: StudentObservation) -> float:
    if observation.outcome == "correct":
        likelihood = 0.05 + point
    elif observation.outcome == "incorrect":
        likelihood = 0.05 + (1.0 - point)
    elif observation.outcome == "partial":
        likelihood = 0.05 + max(0.0, 1.0 - abs(point - 0.5) * 2.0)
    else:
        likelihood = 0.05 + (1.0 - point) ** 1.5

    predicted_time = 6.0 + (1.0 - point) * 20.0
    timing_penalty = min(1.0, abs(observation.response_time_seconds - predicted_time) / 30.0)
    likelihood *= 1.0 - (timing_penalty * 0.35)

    if observation.hint_used:
        likelihood *= 1.0 + max(0.0, 0.5 - point)
    else:
        likelihood *= 1.0 + max(0.0, point - 0.5) * 0.25

    return max(1e-5, likelihood)


def _posterior_mean(prior_mean: float, confidence: float, observation: StudentObservation) -> float:
    spread = max(0.08, 0.35 - confidence * 0.22)
    unnormalized: list[float] = []
    for point in GRID:
        prior_weight = math.exp(-((point - prior_mean) ** 2) / (2 * spread**2))
        unnormalized.append(prior_weight * _observation_likelihood(point, observation))

    normalizer = sum(unnormalized)
    if normalizer <= 0:
        return prior_mean

    return clamp(sum(point * weight for point, weight in zip(GRID, unnormalized)) / normalizer)


def update_belief(
    belief: BeliefConceptState,
    observation: StudentObservation,
    session_elapsed_seconds: float,
) -> StudentObservation:
    """Run a single concept update and adapt the decay rate if prediction was wrong."""

    prior_mean = belief.estimated_mastery
    predicted_correct = prior_mean >= ASSESSMENT_READY_THRESHOLD
    posterior_mean = _posterior_mean(prior_mean, belief.confidence, observation)

    belief.estimated_mastery = posterior_mean
    belief.confidence = clamp(belief.confidence + 0.08, 0.05, 0.98)
    belief.observation_count += 1
    belief.last_observed_at_seconds = session_elapsed_seconds

    adaptation_rate = BASE_ADAPTATION_RATE / (1 + belief.observation_count * ADAPTATION_ALPHA)
    if observation.outcome == "correct" and not predicted_correct:
        belief.decay_rate *= 1.0 - adaptation_rate
    elif observation.outcome in {"incorrect", "skipped"} and predicted_correct:
        belief.decay_rate *= 1.0 + adaptation_rate
    elif observation.outcome == "partial":
        belief.decay_rate *= 1.0 + (adaptation_rate * 0.2)

    belief.decay_rate = max(0.05, min(0.55, belief.decay_rate))
    observation.predicted_correct = predicted_correct
    return observation
