"""Scenario for Task 1: core review scheduling."""

from __future__ import annotations

from adaptive_learning_system.config.defaults import TASK_REGISTRY
from adaptive_learning_system.env.types import ScenarioDefinition

from .shared import belief, build_topic, hidden, registry


def build_task() -> ScenarioDefinition:
    """Create the base algebra review scenario."""

    nodes = build_topic("algebra")
    hidden_state = {
        "number_sense": hidden(0.90, 10, 8),
        "fractions": hidden(0.68, 20, 5),
        "linear_equations": hidden(0.52, 30, 4),
        "inequalities": hidden(0.32, 22, 2),
        "quadratics": hidden(0.00, 0, 0),
        "graph_reading": hidden(0.47, 18, 2),
    }
    belief_state = {
        "number_sense": belief(0.88, 0.90, observation_count=8),
        "fractions": belief(0.64, 0.72, observation_count=5),
        "linear_equations": belief(0.55, 0.60, observation_count=4),
        "inequalities": belief(0.29, 0.48, observation_count=2),
        "quadratics": belief(0.00, 0.12),
        "graph_reading": belief(0.42, 0.45, observation_count=2),
    }
    return ScenarioDefinition(
        name="task1_review",
        description=TASK_REGISTRY["task1_review"]["description"],
        step_budget=TASK_REGISTRY["task1_review"]["step_budget"],
        concept_nodes=nodes,
        hidden_state=hidden_state,
        belief_state=belief_state,
        topic_registry=registry("calculus", "geometry"),
    )
