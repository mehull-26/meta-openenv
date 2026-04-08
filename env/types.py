"""Internal dataclasses used by the environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Outcome = Literal["correct", "incorrect", "partial", "skipped"]
ActionType = Literal["Review", "Introduce", "Assess", "AddSubject", "Wait"]


@dataclass(slots=True)
class ConceptNode:
    concept_id: str
    topic: str
    difficulty: float
    content_ref: str
    prerequisites: tuple[str, ...] = ()


@dataclass(slots=True)
class HiddenConceptState:
    true_mastery: float
    time_since_last_seconds: float
    exposure_count: int
    stage: str
    consecutive_incorrect: int = 0


@dataclass(slots=True)
class BeliefConceptState:
    estimated_mastery: float
    confidence: float
    decay_rate: float
    observation_count: int
    last_observed_at_seconds: float | None = None


@dataclass(slots=True)
class StudentObservation:
    concept_id: str
    outcome: Outcome
    response_time_seconds: float
    hint_used: bool
    predicted_correct: bool | None = None
    gap_seconds: float = 0.0


@dataclass(slots=True)
class TopicSignal:
    topic_name: str
    assessment_score: float
    reason: str


@dataclass(slots=True)
class ScenarioDefinition:
    name: str
    description: str
    step_budget: int
    concept_nodes: list[ConceptNode]
    hidden_state: dict[str, HiddenConceptState]
    belief_state: dict[str, BeliefConceptState]
    topic_registry: dict[str, list[ConceptNode]]
    pending_topics: list[TopicSignal] = field(default_factory=list)
    topic_unlock_schedule: dict[int, list[TopicSignal]] = field(default_factory=dict)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a float into an inclusive range."""

    return max(minimum, min(maximum, value))
