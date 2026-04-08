"""Shared scenario helpers."""

from __future__ import annotations

from adaptive_learning_system.config.defaults import BASE_DECAY_RATE, TOPIC_LIBRARY
from adaptive_learning_system.env.hidden_state import mastery_to_stage
from adaptive_learning_system.env.types import (
    BeliefConceptState,
    ConceptNode,
    HiddenConceptState,
)


def build_topic(topic_name: str) -> list[ConceptNode]:
    """Clone a topic definition into runtime nodes."""

    return [
        ConceptNode(
            concept_id=spec["concept_id"],
            topic=topic_name,
            difficulty=spec["difficulty"],
            content_ref=spec["content_ref"],
            prerequisites=tuple(spec["prerequisites"]),
        )
        for spec in TOPIC_LIBRARY[topic_name]
    ]


def registry(*topic_names: str) -> dict[str, list[ConceptNode]]:
    """Create a small topic registry for AddSubject."""

    return {topic_name: build_topic(topic_name) for topic_name in topic_names}


def hidden(
    mastery: float,
    hours_since_last: float,
    exposure_count: int,
    consecutive_incorrect: int = 0,
) -> HiddenConceptState:
    """Build hidden concept state using readable units in task definitions."""

    return HiddenConceptState(
        true_mastery=mastery,
        time_since_last_seconds=hours_since_last * 3600,
        exposure_count=exposure_count,
        stage=mastery_to_stage(mastery),
        consecutive_incorrect=consecutive_incorrect,
    )


def belief(
    mastery: float,
    confidence: float,
    decay_rate: float = BASE_DECAY_RATE,
    observation_count: int = 0,
) -> BeliefConceptState:
    """Build belief state for a concept."""

    return BeliefConceptState(
        estimated_mastery=mastery,
        confidence=confidence,
        decay_rate=decay_rate,
        observation_count=observation_count,
        last_observed_at_seconds=0.0 if observation_count else None,
    )
