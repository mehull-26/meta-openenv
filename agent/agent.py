"""Thin policy helpers that rank only valid environment actions."""

from __future__ import annotations

from adaptive_learning_system.config.defaults import RETENTION_THRESHOLD
from adaptive_learning_system.models import AdaptiveLearningSystemObservation


def _action_concept_id(action: str) -> str | None:
    if "(" not in action or not action.endswith(")"):
        return None
    kind, payload = action.split("(", 1)
    if kind == "AddSubject":
        return None
    return payload[:-1]


def _concept_metrics(observation: AdaptiveLearningSystemObservation, concept_id: str) -> tuple[float, float, float]:
    for concept in observation.belief_snapshot:
        if concept.concept_id == concept_id:
            return concept.estimated_mastery, concept.urgency, concept.confidence
    return 1.0, -1.0, 1.0


def _sort_review_actions(
    observation: AdaptiveLearningSystemObservation,
    actions: list[str],
) -> list[str]:
    return sorted(
        actions,
        key=lambda action: (
            _concept_metrics(observation, _action_concept_id(action) or "")[0],
            -_concept_metrics(observation, _action_concept_id(action) or "")[1],
            _action_concept_id(action) or action,
        ),
    )


def _sort_assess_actions(
    observation: AdaptiveLearningSystemObservation,
    actions: list[str],
) -> list[str]:
    return sorted(
        actions,
        key=lambda action: (
            -_concept_metrics(observation, _action_concept_id(action) or "")[1],
            _concept_metrics(observation, _action_concept_id(action) or "")[2],
            _action_concept_id(action) or action,
        ),
    )


def choose_best_available_action(observation: AdaptiveLearningSystemObservation) -> str:
    """Choose a sensible valid action directly from available_actions."""

    available_actions = list(observation.available_actions)
    if not available_actions:
        return "Wait"

    add_subject_actions = sorted(
        action for action in available_actions if action.startswith("AddSubject(")
    )
    introduce_actions = sorted(
        action for action in available_actions if action.startswith("Introduce(")
    )
    review_actions = [
        action for action in available_actions if action.startswith("Review(")
    ]
    assess_actions = [
        action for action in available_actions if action.startswith("Assess(")
    ]

    urgent_review_actions = [
        action
        for action in review_actions
        if _concept_metrics(observation, _action_concept_id(action) or "")[0] < RETENTION_THRESHOLD
    ]
    if urgent_review_actions:
        return _sort_review_actions(observation, urgent_review_actions)[0]

    if (
        add_subject_actions
        and observation.aggregate_stats.below_threshold == 0
        and observation.aggregate_stats.average_mastery >= 0.68
    ):
        return add_subject_actions[0]

    if (
        introduce_actions
        and observation.aggregate_stats.below_threshold == 0
        and observation.aggregate_stats.average_mastery >= 0.62
    ):
        return introduce_actions[0]

    if assess_actions:
        return _sort_assess_actions(observation, assess_actions)[0]

    if review_actions:
        return _sort_review_actions(observation, review_actions)[0]

    if introduce_actions:
        return introduce_actions[0]

    return available_actions[0] if available_actions else "Wait"


class AdaptiveLearningPolicyAgent:
    """Simple heuristic policy that mirrors the architecture's intended behavior."""

    def choose_action(self, observation: AdaptiveLearningSystemObservation) -> str:
        return choose_best_available_action(observation)
