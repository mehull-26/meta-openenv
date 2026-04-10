"""Reward computation grounded in productive learning utility."""

from __future__ import annotations

import math

from adaptive_learning_system.config.defaults import RETENTION_THRESHOLD, REWARD_WEIGHTS

from .types import BeliefConceptState, HiddenConceptState, StudentObservation, clamp


def _gap_factor(gap_seconds: float) -> float:
    """Scale retention-sensitive bonuses by how long the concept has gone untouched."""

    return clamp(gap_seconds / (18 * 60 * 60))


def _bounded_reward(positive_utility: float, negative_utility: float) -> float:
    """Map unbounded utilities into [0, 1] with near-zero neutral reward."""

    floor = REWARD_WEIGHTS["reward_floor"]
    productive_term = 1.0 - math.exp(-max(0.0, positive_utility))
    risk_discount = math.exp(-max(0.0, negative_utility))
    return clamp(risk_discount * (floor + ((1.0 - floor) * productive_term)))


def compute_reward(
    previous_beliefs: dict[str, float],
    beliefs: dict[str, BeliefConceptState],
    hidden_state: dict[str, HiddenConceptState],
    observation: StudentObservation | None,
    action_type: str,
    invalid_action: bool = False,
) -> tuple[float, list[str]]:
    """Compute a bounded step reward from productive learning and opportunity cost."""

    notes: list[str] = []
    positive_utility = 0.0
    negative_utility = 0.0

    mastery_gain = 0.0
    mastery_loss = 0.0
    threshold_recoveries: list[str] = []
    threshold_drops: list[str] = []

    for concept_id, belief in beliefs.items():
        before = previous_beliefs.get(concept_id, belief.estimated_mastery)
        after = belief.estimated_mastery
        mastery_gain += max(0.0, after - before)
        mastery_loss += max(0.0, before - after)
        if before < RETENTION_THRESHOLD <= after:
            threshold_recoveries.append(concept_id)
        elif before >= RETENTION_THRESHOLD > after:
            threshold_drops.append(concept_id)

    if mastery_gain > 0:
        positive_utility += mastery_gain * REWARD_WEIGHTS["mastery_gain"]
        notes.append(f"Belief mastery improved by {mastery_gain:.2f}.")

    if mastery_loss > 0:
        negative_utility += mastery_loss * REWARD_WEIGHTS["mastery_loss"]
        notes.append(f"Belief decay cost {mastery_loss:.2f}.")

    if threshold_recoveries:
        positive_utility += len(threshold_recoveries) * REWARD_WEIGHTS["threshold_recovery"]
        notes.append(
            "Recovered above retention threshold: "
            + ", ".join(threshold_recoveries[:3])
            + ("..." if len(threshold_recoveries) > 3 else "")
            + "."
        )

    if threshold_drops:
        negative_utility += len(threshold_drops) * REWARD_WEIGHTS["threshold_drop"]
        notes.append(
            "Dropped below retention threshold: "
            + ", ".join(threshold_drops[:3])
            + ("..." if len(threshold_drops) > 3 else "")
            + "."
        )

    total_concepts = max(len(beliefs), 1)
    below_threshold_ratio = (
        sum(1 for belief in beliefs.values() if belief.estimated_mastery < RETENTION_THRESHOLD)
        / total_concepts
    )

    if observation:
        gap_factor = _gap_factor(observation.gap_seconds)
        if observation.outcome == "correct":
            positive_utility += REWARD_WEIGHTS["correct_bonus"] * (1.0 + gap_factor)
            notes.append("Correct response strengthened evidence of retention.")
            if observation.gap_seconds >= 12 * 60 * 60:
                positive_utility += REWARD_WEIGHTS["retention_bonus"]
                notes.append("Correct after a long gap earned a retention bonus.")
        elif observation.outcome == "partial":
            positive_utility += REWARD_WEIGHTS["partial_bonus"] * (1.0 + 0.5 * gap_factor)
            notes.append("Partial response still produced useful learning signal.")
        elif observation.outcome == "incorrect":
            negative_utility += REWARD_WEIGHTS["incorrect_penalty"] * (1.0 + 0.5 * gap_factor)
            notes.append("Incorrect response spent a turn without improving retention.")
        elif observation.outcome == "skipped":
            negative_utility += REWARD_WEIGHTS["skipped_penalty"] * (1.0 + 0.5 * gap_factor)
            notes.append("Skipped response indicates low engagement on the chosen concept.")

        hidden = hidden_state[observation.concept_id]
        if hidden.consecutive_incorrect >= 3:
            negative_utility += REWARD_WEIGHTS["frustration_penalty"]
            notes.append("Repeated incorrect answers increased frustration risk.")

    # Assess flows through the observation path above (correct/partial/incorrect/skipped)
    # without a named bonus — calibration value is captured through the belief update.

    if action_type == "Introduce" and observation and observation.outcome in {"correct", "partial"}:
        multiplier = 1.0 if observation.outcome == "correct" else 0.5
        positive_utility += REWARD_WEIGHTS["exploration_bonus"] * multiplier
        notes.append("Successful concept introduction added exploration value.")

    if action_type == "AddSubject" and len(beliefs) > len(previous_beliefs):
        stability = 1.0 - below_threshold_ratio
        positive_utility += REWARD_WEIGHTS["subject_expansion_bonus"] * (0.5 + 0.5 * stability)
        notes.append("Expanded the curriculum while preserving prior progress.")

    if action_type == "Wait":
        negative_utility += REWARD_WEIGHTS["wait_penalty"] + (
            below_threshold_ratio * REWARD_WEIGHTS["unmet_need_penalty"]
        )
        notes.append("Waiting consumed a turn while learning needs remained unresolved.")

    if invalid_action:
        negative_utility += REWARD_WEIGHTS["invalid_action_penalty"]
        notes.append("Invalid action was filtered to Wait.")

    reward = _bounded_reward(positive_utility, negative_utility)
    return reward, notes
