"""Scenario for Task 3: triage under pressure."""

from __future__ import annotations

from adaptive_learning_system.config.defaults import TASK_REGISTRY
from adaptive_learning_system.env.types import ScenarioDefinition

from .shared import belief, build_topic, hidden, registry


def build_task() -> ScenarioDefinition:
    """Create the high-pressure retention scenario."""

    nodes = build_topic("algebra") + build_topic("geometry")
    hidden_state = {
        "number_sense": hidden(0.88, 14, 8),
        "fractions": hidden(0.44, 30, 5, consecutive_incorrect=1),
        "linear_equations": hidden(0.37, 40, 5, consecutive_incorrect=2),
        "inequalities": hidden(0.28, 28, 3),
        "quadratics": hidden(0.21, 24, 2),
        "graph_reading": hidden(0.41, 26, 3),
        "angle_basics": hidden(0.49, 20, 3),
        "triangles": hidden(0.31, 18, 2),
        "coordinate_geometry": hidden(0.00, 0, 0),
    }
    belief_state = {
        "number_sense": belief(0.85, 0.88, observation_count=8),
        "fractions": belief(0.47, 0.68, observation_count=5),
        "linear_equations": belief(0.42, 0.62, observation_count=5),
        "inequalities": belief(0.31, 0.52, observation_count=3),
        "quadratics": belief(0.24, 0.40, observation_count=2),
        "graph_reading": belief(0.45, 0.50, observation_count=3),
        "angle_basics": belief(0.52, 0.55, observation_count=3),
        "triangles": belief(0.33, 0.38, observation_count=2),
        "coordinate_geometry": belief(0.00, 0.08),
    }
    return ScenarioDefinition(
        name="task3_triage",
        description=TASK_REGISTRY["task3_triage"]["description"],
        step_budget=TASK_REGISTRY["task3_triage"]["step_budget"],
        concept_nodes=nodes,
        hidden_state=hidden_state,
        belief_state=belief_state,
        topic_registry=registry("calculus"),
    )
