"""Scenario for Task 2: adding a new subject mid-session."""

from __future__ import annotations

from adaptive_learning_system.config.defaults import TASK_REGISTRY
from adaptive_learning_system.env.types import ScenarioDefinition, TopicSignal

from .shared import belief, build_topic, hidden, registry


def build_task() -> ScenarioDefinition:
    """Create a scenario where calculus becomes available after stabilization."""

    nodes = build_topic("algebra")
    hidden_state = {
        "number_sense": hidden(0.94, 8, 10),
        "fractions": hidden(0.82, 16, 7),
        "linear_equations": hidden(0.78, 18, 6),
        "inequalities": hidden(0.58, 20, 4),
        "quadratics": hidden(0.36, 12, 2),
        "graph_reading": hidden(0.73, 10, 5),
    }
    belief_state = {
        "number_sense": belief(0.92, 0.95, observation_count=10),
        "fractions": belief(0.80, 0.85, observation_count=7),
        "linear_equations": belief(0.76, 0.80, observation_count=6),
        "inequalities": belief(0.54, 0.62, observation_count=4),
        "quadratics": belief(0.32, 0.40, observation_count=2),
        "graph_reading": belief(0.70, 0.76, observation_count=5),
    }
    return ScenarioDefinition(
        name="task2_add_subject",
        description=TASK_REGISTRY["task2_add_subject"]["description"],
        step_budget=TASK_REGISTRY["task2_add_subject"]["step_budget"],
        concept_nodes=nodes,
        hidden_state=hidden_state,
        belief_state=belief_state,
        topic_registry=registry("calculus", "geometry"),
        topic_unlock_schedule={
            4: [
                TopicSignal(
                    topic_name="calculus",
                    assessment_score=0.78,
                    reason="Exploratory assessment suggests readiness for calculus.",
                )
            ]
        },
    )
