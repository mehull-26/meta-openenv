"""Pydantic models exposed through the OpenEnv API."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator


ActionType = Literal["Review", "Introduce", "Assess", "AddSubject", "Wait"]
OutcomeType = Literal["correct", "incorrect", "partial", "skipped"]


class ConceptSnapshot(BaseModel):
    """Compact concept view shown to the policy."""

    concept_id: str
    topic: str
    stage: str
    estimated_mastery: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: float = Field(..., ge=0.0)
    time_since_review_hours: float = Field(..., ge=0.0)
    prerequisites_met: bool = True
    consecutive_incorrect: int = Field(default=0, ge=0)


class AggregateStats(BaseModel):
    """High-level environment stats included in each summary."""

    total_concepts: int = Field(..., ge=0)
    below_threshold: int = Field(..., ge=0)
    average_mastery: float = Field(..., ge=0.0, le=1.0)
    session_hours: float = Field(..., ge=0.0)
    unlockable_topics: list[str] = Field(default_factory=list)
    concepts_with_incorrect_streak: int = Field(default=0, ge=0)
    frustration_risk_count: int = Field(default=0, ge=0)


class PendingTopicSummary(BaseModel):
    """New topic signal exposed when AddSubject becomes available."""

    topic_name: str
    assessment_score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class ObservationDetails(BaseModel):
    """Student interaction sampled by the environment."""

    concept_id: str
    outcome: OutcomeType
    response_time_seconds: float = Field(..., ge=0.0)
    hint_used: bool = False
    predicted_correct: bool | None = None


class AdaptiveLearningSystemAction(Action):
    """Single action selected by the agent or policy."""

    action_type: ActionType = Field(..., description="Chosen action family.")
    concept_id: str | None = Field(
        default=None,
        description="Concept identifier for Review, Introduce, or Assess.",
    )
    topic_name: str | None = Field(
        default=None,
        description="Topic name for AddSubject actions.",
    )

    @model_validator(mode="after")
    def validate_parameters(self) -> "AdaptiveLearningSystemAction":
        if self.action_type in {"Review", "Introduce", "Assess"} and not self.concept_id:
            raise ValueError(f"{self.action_type} requires concept_id")
        if self.action_type == "AddSubject" and not self.topic_name:
            raise ValueError("AddSubject requires topic_name")
        return self


class AdaptiveLearningSystemObservation(Observation):
    """Summary emitted after reset and every environment step."""

    task_name: str = ""
    summary_message: str = ""
    chosen_action: str = ""
    belief_snapshot: list[ConceptSnapshot] = Field(default_factory=list)
    aggregate_stats: AggregateStats = Field(
        default_factory=lambda: AggregateStats(
            total_concepts=0,
            below_threshold=0,
            average_mastery=0.0,
            session_hours=0.0,
            unlockable_topics=[],
            concepts_with_incorrect_streak=0,
            frustration_risk_count=0,
        )
    )
    last_observation: ObservationDetails | None = None
    available_actions: list[str] = Field(default_factory=list)
    pending_topics: list[PendingTopicSummary] = Field(default_factory=list)
    graph_context: str = ""
    recommended_action: str = ""
    notes: list[str] = Field(default_factory=list)
    last_action_error: str | None = None
    grader_score: float = Field(default=0.0, ge=0.0, le=1.0)
    task_metrics: dict[str, float] = Field(default_factory=dict)
