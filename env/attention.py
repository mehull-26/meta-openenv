"""Attentional filter used to keep the LLM summary compact."""

from __future__ import annotations

from adaptive_learning_system.config.defaults import ATTENTION_SLOTS, LEARNED_THRESHOLD, RETENTION_THRESHOLD
from adaptive_learning_system.models import AggregateStats, ConceptSnapshot

from .concept_graph import ConceptGraph
from .hidden_state import mastery_to_stage
from .types import BeliefConceptState, HiddenConceptState, StudentObservation


def _time_since_review_hours(
    session_elapsed_seconds: float,
    belief: BeliefConceptState,
) -> float:
    if belief.last_observed_at_seconds is None:
        return session_elapsed_seconds / 3600
    return max(0.0, (session_elapsed_seconds - belief.last_observed_at_seconds) / 3600)


def _urgency(
    session_elapsed_seconds: float,
    belief: BeliefConceptState,
) -> float:
    hours = _time_since_review_hours(session_elapsed_seconds, belief)
    return (1.0 - belief.estimated_mastery) + min(1.5, hours / 24.0)


def build_attention_summary(
    graph: ConceptGraph,
    beliefs: dict[str, BeliefConceptState],
    hidden_state: dict[str, HiddenConceptState],
    session_elapsed_seconds: float,
    last_observation: StudentObservation | None,
    unlockable_topics: list[str],
) -> tuple[list[ConceptSnapshot], AggregateStats]:
    """Select the fixed-size concept window described in the architecture."""

    selected: list[str] = []
    seen: set[str] = set()

    ranked = sorted(
        graph.all_nodes(),
        key=lambda node: _urgency(session_elapsed_seconds, beliefs[node.concept_id]),
        reverse=True,
    )
    for node in ranked[: ATTENTION_SLOTS["critical"]]:
        selected.append(node.concept_id)
        seen.add(node.concept_id)

    frontier = [
        node
        for node in graph.all_nodes()
        if mastery_to_stage(beliefs[node.concept_id].estimated_mastery) == "learning"
        and node.concept_id not in seen
    ]
    for node in frontier[: ATTENTION_SLOTS["frontier"]]:
        selected.append(node.concept_id)
        seen.add(node.concept_id)

    ready = [
        node
        for node in graph.ready_to_introduce(beliefs, LEARNED_THRESHOLD)
        if node.concept_id not in seen
    ]
    for node in ready[: ATTENTION_SLOTS["ready"]]:
        selected.append(node.concept_id)
        seen.add(node.concept_id)

    if last_observation and last_observation.concept_id not in seen:
        selected.append(last_observation.concept_id)

    snapshot = [
        ConceptSnapshot(
            concept_id=concept_id,
            topic=graph.get(concept_id).topic,
            stage=mastery_to_stage(beliefs[concept_id].estimated_mastery),
            estimated_mastery=beliefs[concept_id].estimated_mastery,
            confidence=beliefs[concept_id].confidence,
            urgency=_urgency(session_elapsed_seconds, beliefs[concept_id]),
            time_since_review_hours=_time_since_review_hours(
                session_elapsed_seconds,
                beliefs[concept_id],
            ),
            prerequisites_met=graph.prerequisites_met(
                concept_id,
                beliefs,
                LEARNED_THRESHOLD,
            ),
            consecutive_incorrect=hidden_state[concept_id].consecutive_incorrect,
        )
        for concept_id in selected
    ]

    average_mastery = sum(belief.estimated_mastery for belief in beliefs.values()) / max(len(beliefs), 1)
    aggregate = AggregateStats(
        total_concepts=len(beliefs),
        below_threshold=sum(
            1 for belief in beliefs.values() if belief.estimated_mastery < RETENTION_THRESHOLD
        ),
        average_mastery=average_mastery,
        session_hours=session_elapsed_seconds / 3600,
        unlockable_topics=unlockable_topics,
        concepts_with_incorrect_streak=sum(
            1 for state in hidden_state.values() if state.consecutive_incorrect > 0
        ),
        frustration_risk_count=sum(
            1 for state in hidden_state.values() if state.consecutive_incorrect >= 2
        ),
    )
    return snapshot, aggregate
