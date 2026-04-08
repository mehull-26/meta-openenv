"""Lightweight regression tests for core environment behavior."""

from __future__ import annotations

import unittest

from adaptive_learning_system.config.defaults import LEARNED_THRESHOLD
from adaptive_learning_system.env.environment import AdaptiveLearningRuntime
from adaptive_learning_system.models import AdaptiveLearningSystemAction


TASKS = ("task1_review", "task2_add_subject", "task3_triage")


def _run_wait_policy(task_name: str) -> tuple[list[float], object]:
    runtime = AdaptiveLearningRuntime(task_name=task_name, seed=7)
    observation = runtime.reset(task_name=task_name)
    rewards: list[float] = []
    while not observation.done:
        observation = runtime.step(AdaptiveLearningSystemAction(action_type="Wait"))
        rewards.append(float(observation.reward or 0.0))
    return rewards, observation


class RuntimeSanityTests(unittest.TestCase):
    def test_reset_reproduces_fixed_action_trace(self) -> None:
        runtime = AdaptiveLearningRuntime(task_name="task1_review", seed=7)
        actions = [
            AdaptiveLearningSystemAction(action_type="Review", concept_id="fractions"),
            AdaptiveLearningSystemAction(action_type="Review", concept_id="linear_equations"),
            AdaptiveLearningSystemAction(action_type="Assess", concept_id="number_sense"),
            AdaptiveLearningSystemAction(action_type="Wait"),
        ]

        def trace_once() -> list[tuple[str | None, float, float, str | None]]:
            runtime.reset(task_name="task1_review")
            trace: list[tuple[str | None, float, float, str | None]] = []
            for action in actions:
                observation = runtime.step(action)
                trace.append(
                    (
                        observation.last_observation.outcome if observation.last_observation else None,
                        round(float(observation.reward or 0.0), 6),
                        round(observation.grader_score, 6),
                        observation.last_action_error,
                    )
                )
            return trace

        self.assertEqual(trace_once(), trace_once())

    def test_recommended_action_is_valid_for_all_tasks(self) -> None:
        for task_name in TASKS:
            runtime = AdaptiveLearningRuntime(task_name=task_name, seed=7)
            observation = runtime.reset(task_name=task_name)
            self.assertIn(observation.recommended_action, observation.available_actions)

    def test_wait_policy_reward_stays_near_zero(self) -> None:
        for task_name in TASKS:
            rewards, _ = _run_wait_policy(task_name)
            self.assertTrue(rewards)
            self.assertLess(sum(rewards) / len(rewards), 0.05)

    def test_task1_concept_graph_gates_introductions_by_prerequisites(self) -> None:
        runtime = AdaptiveLearningRuntime(task_name="task1_review", seed=7)
        observation = runtime.reset(task_name="task1_review")

        self.assertIn("Introduce(quadratics)", observation.available_actions)
        self.assertEqual(
            runtime._graph.get("quadratics").prerequisites,
            ("linear_equations",),
        )
        self.assertTrue(
            runtime._graph.prerequisites_met(
                "quadratics",
                runtime._belief_state,
                LEARNED_THRESHOLD,
            )
        )

    def test_task2_add_subject_merges_concept_subgraph(self) -> None:
        runtime = AdaptiveLearningRuntime(task_name="task2_add_subject", seed=7)
        observation = runtime.reset(task_name="task2_add_subject")

        for _ in range(4):
            observation = runtime.step(AdaptiveLearningSystemAction(action_type="Wait"))

        self.assertIn("AddSubject(calculus)", observation.available_actions)
        self.assertEqual(len(observation.pending_topics), 1)
        self.assertEqual(observation.pending_topics[0].topic_name, "calculus")
        self.assertAlmostEqual(observation.pending_topics[0].assessment_score, 0.78, places=2)
        observation = runtime.step(
            AdaptiveLearningSystemAction(action_type="AddSubject", topic_name="calculus")
        )

        for concept_id in ("rate_of_change", "limits", "derivatives"):
            self.assertEqual(runtime._graph.get(concept_id).topic, "calculus")

        self.assertIn("Review(rate_of_change)", observation.available_actions)
        self.assertTrue(
            runtime._graph.prerequisites_met(
                "rate_of_change",
                runtime._belief_state,
                LEARNED_THRESHOLD,
            )
        )
        self.assertFalse(
            runtime._graph.prerequisites_met(
                "derivatives",
                runtime._belief_state,
                LEARNED_THRESHOLD,
            )
        )
        self.assertEqual(observation.pending_topics, [])

    def test_task2_grader_requires_actual_subject_expansion(self) -> None:
        _, wait_observation = _run_wait_policy("task2_add_subject")

        runtime = AdaptiveLearningRuntime(task_name="task2_add_subject", seed=7)
        observation = runtime.reset(task_name="task2_add_subject")
        for _ in range(4):
            observation = runtime.step(AdaptiveLearningSystemAction(action_type="Wait"))

        self.assertIn("AddSubject(calculus)", observation.available_actions)
        observation = runtime.step(
            AdaptiveLearningSystemAction(action_type="AddSubject", topic_name="calculus")
        )
        while not observation.done:
            observation = runtime.step(AdaptiveLearningSystemAction(action_type="Wait"))

        self.assertEqual(wait_observation.grader_score, 0.0)
        self.assertEqual(wait_observation.task_metrics["introduced_ratio"], 0.0)
        self.assertGreater(observation.grader_score, wait_observation.grader_score)
        self.assertEqual(observation.task_metrics["introduced_ratio"], 1.0)

    def test_task3_grader_penalizes_strong_concept_spam(self) -> None:
        runtime = AdaptiveLearningRuntime(task_name="task3_triage", seed=7)
        observation = runtime.reset(task_name="task3_triage")
        spam_action = AdaptiveLearningSystemAction(action_type="Assess", concept_id="number_sense")

        while not observation.done:
            observation = runtime.step(spam_action)

        self.assertLess(observation.grader_score, 0.1)
        self.assertEqual(observation.task_metrics["rescue_gain_ratio"], 0.0)
        self.assertGreaterEqual(observation.task_metrics["unresolved_risk_ratio"], 0.9)

    def test_task3_observation_exposes_frustration_signals(self) -> None:
        runtime = AdaptiveLearningRuntime(task_name="task3_triage", seed=7)
        observation = runtime.reset(task_name="task3_triage")

        self.assertEqual(observation.aggregate_stats.concepts_with_incorrect_streak, 2)
        self.assertEqual(observation.aggregate_stats.frustration_risk_count, 1)
        self.assertTrue(
            any(snapshot.consecutive_incorrect > 0 for snapshot in observation.belief_snapshot)
        )


if __name__ == "__main__":
    unittest.main()
