"""Task graders used for scoring baseline runs."""

from __future__ import annotations

from adaptive_learning_system.config.defaults import RETENTION_THRESHOLD
from adaptive_learning_system.env.concept_graph import ConceptGraph
from adaptive_learning_system.env.types import (
    BeliefConceptState,
    HiddenConceptState,
    ScenarioDefinition,
    clamp,
)

MIN_SUBMISSION_SCORE = 0.001
MAX_SUBMISSION_SCORE = 0.999


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalized_progress(belief: BeliefConceptState | None, target_mastery: float) -> float:
    if belief is None or target_mastery <= 0:
        return 0.0
    return clamp(belief.estimated_mastery / target_mastery)


def _submission_score(value: float) -> float:
    """Keep task scores strictly inside (0, 1) after three-decimal log formatting."""

    return clamp(value, minimum=MIN_SUBMISSION_SCORE, maximum=MAX_SUBMISSION_SCORE)


def grade_task1_review(
    beliefs: dict[str, BeliefConceptState],
    hidden_state: dict[str, HiddenConceptState],
    graph: ConceptGraph,
    scenario: ScenarioDefinition,
) -> tuple[float, dict[str, float]]:
    """Score mastery preservation for the core review task."""

    del hidden_state, graph, scenario
    total = max(len(beliefs), 1)
    average_mastery = sum(belief.estimated_mastery for belief in beliefs.values()) / total
    retained_ratio = (
        sum(1 for belief in beliefs.values() if belief.estimated_mastery >= RETENTION_THRESHOLD) / total
    )
    score = _submission_score((0.7 * average_mastery) + (0.3 * retained_ratio))
    return score, {
        "average_mastery": round(average_mastery, 4),
        "retained_ratio": round(retained_ratio, 4),
    }


def grade_task2_add_subject(
    beliefs: dict[str, BeliefConceptState],
    hidden_state: dict[str, HiddenConceptState],
    graph: ConceptGraph,
    scenario: ScenarioDefinition,
) -> tuple[float, dict[str, float]]:
    """Score calculus expansion only when prior algebra mastery remains protected."""

    del hidden_state, graph
    protected_concepts = tuple(scenario.belief_state.keys())
    expansion_concepts = tuple(
        node.concept_id for node in scenario.topic_registry.get("calculus", [])
    )
    initially_stable_concepts = tuple(
        concept_id
        for concept_id in protected_concepts
        if scenario.belief_state[concept_id].estimated_mastery >= RETENTION_THRESHOLD
    )

    protected_retained_ratio = (
        sum(
            1
            for concept_id in protected_concepts
            if beliefs[concept_id].estimated_mastery >= RETENTION_THRESHOLD
        )
        / max(len(protected_concepts), 1)
    )
    protected_preservation_ratio = _mean(
        [
            min(
                1.0,
                beliefs[concept_id].estimated_mastery
                / max(scenario.belief_state[concept_id].estimated_mastery, 0.01),
            )
            for concept_id in protected_concepts
        ]
    )
    collapse_ratio = (
        sum(
            1
            for concept_id in initially_stable_concepts
            if beliefs[concept_id].estimated_mastery < RETENTION_THRESHOLD
        )
        / max(len(initially_stable_concepts), 1)
    )
    introduced_ratio = (
        sum(1 for concept_id in expansion_concepts if concept_id in beliefs)
        / max(len(expansion_concepts), 1)
    )

    rate_of_change_progress = _normalized_progress(beliefs.get("rate_of_change"), 0.55)
    limits_gate = _normalized_progress(beliefs.get("rate_of_change"), 0.45)
    limits_progress = _normalized_progress(beliefs.get("limits"), 0.45) * limits_gate

    quadratics_support = _normalized_progress(
        beliefs.get("quadratics", scenario.belief_state.get("quadratics")),
        RETENTION_THRESHOLD,
    )
    derivatives_gate = _normalized_progress(beliefs.get("limits"), 0.40) * quadratics_support
    derivatives_progress = (
        _normalized_progress(beliefs.get("derivatives"), 0.35) * derivatives_gate
    )

    maintenance_score = (
        (0.60 * protected_retained_ratio) + (0.40 * protected_preservation_ratio)
    )
    expansion_score = (
        (0.30 * introduced_ratio)
        + (0.45 * _mean([rate_of_change_progress, limits_progress]))
        + (0.25 * derivatives_progress)
    )
    score = _submission_score(
        introduced_ratio
        * ((0.60 * maintenance_score) + (0.40 * expansion_score))
        * (1.0 - (0.75 * collapse_ratio))
    )
    return score, {
        "protected_retained_ratio": round(protected_retained_ratio, 4),
        "protected_preservation_ratio": round(protected_preservation_ratio, 4),
        "collapse_ratio": round(collapse_ratio, 4),
        "introduced_ratio": round(introduced_ratio, 4),
        "rate_of_change_progress": round(rate_of_change_progress, 4),
        "limits_progress": round(limits_progress, 4),
        "derivatives_progress": round(derivatives_progress, 4),
        "maintenance_score": round(maintenance_score, 4),
        "expansion_score": round(expansion_score, 4),
    }


def grade_task3_triage(
    beliefs: dict[str, BeliefConceptState],
    hidden_state: dict[str, HiddenConceptState],
    graph: ConceptGraph,
    scenario: ScenarioDefinition,
) -> tuple[float, dict[str, float]]:
    """Score rescue of the initially struggling concepts under time pressure."""

    del graph
    focus_concepts = [
        concept_id
        for concept_id, state in scenario.hidden_state.items()
        if (0.0 < state.true_mastery < RETENTION_THRESHOLD) or state.consecutive_incorrect > 0
    ]
    initial_beliefs = scenario.belief_state
    focus_average_mastery = _mean(
        [beliefs[concept_id].estimated_mastery for concept_id in focus_concepts]
    )
    focus_retained_ratio = (
        sum(
            1
            for concept_id in focus_concepts
            if beliefs[concept_id].estimated_mastery >= RETENTION_THRESHOLD
        )
        / max(len(focus_concepts), 1)
    )
    focus_supported_ratio = (
        sum(
            1
            for concept_id in focus_concepts
            if beliefs[concept_id].estimated_mastery >= 0.40
            and hidden_state[concept_id].consecutive_incorrect < 2
        )
        / max(len(focus_concepts), 1)
    )
    # Normalize to a target average gain of 0.18, calibrated to the gap between the
    # focus concepts' starting average mastery (~0.37) and RETENTION_THRESHOLD (0.55).
    rescue_gain_ratio = clamp(
        _mean(
            [
                max(
                    0.0,
                    beliefs[concept_id].estimated_mastery
                    - initial_beliefs[concept_id].estimated_mastery,
                )
                for concept_id in focus_concepts
            ]
        )
        / 0.18
    )
    unresolved_risk_ratio = (
        sum(
            1
            for concept_id in focus_concepts
            if beliefs[concept_id].estimated_mastery < 0.35
            or hidden_state[concept_id].consecutive_incorrect >= 2
        )
        / max(len(focus_concepts), 1)
    )
    regression_ratio = _mean(
        [
            max(
                0.0,
                initial_beliefs[concept_id].estimated_mastery
                - beliefs[concept_id].estimated_mastery,
            )
            / max(initial_beliefs[concept_id].estimated_mastery, 0.25)
            for concept_id in focus_concepts
        ]
    )

    base_score = (
        (0.35 * focus_average_mastery)
        + (0.25 * focus_supported_ratio)
        + (0.20 * focus_retained_ratio)
        + (0.20 * rescue_gain_ratio)
    )
    risk_discount = (
        (1.0 - (0.65 * unresolved_risk_ratio))
        * (1.0 - (0.50 * regression_ratio))
    )
    score = _submission_score(base_score * max(0.0, risk_discount))
    return score, {
        "focus_average_mastery": round(focus_average_mastery, 4),
        "focus_retained_ratio": round(focus_retained_ratio, 4),
        "focus_supported_ratio": round(focus_supported_ratio, 4),
        "rescue_gain_ratio": round(rescue_gain_ratio, 4),
        "unresolved_risk_ratio": round(unresolved_risk_ratio, 4),
        "regression_ratio": round(regression_ratio, 4),
    }
