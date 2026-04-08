"""Main environment runtime for the adaptive learning system."""

from __future__ import annotations

import copy
import os
import random
from uuid import uuid4

from openenv.core.env_server.types import State

from adaptive_learning_system.agent import AdaptiveLearningPolicyAgent
from adaptive_learning_system.config.defaults import (
    BASE_DECAY_RATE,
    DEFAULT_TASK,
    LEARNED_THRESHOLD,
    STEP_SECONDS,
)
from adaptive_learning_system.models import (
    AdaptiveLearningSystemAction,
    AdaptiveLearningSystemObservation,
    ObservationDetails,
    PendingTopicSummary,
)
from adaptive_learning_system.tasks.grading import (
    grade_task1_review,
    grade_task2_add_subject,
    grade_task3_triage,
)
from adaptive_learning_system.tasks.task1_review import build_task as build_task1_review
from adaptive_learning_system.tasks.task2_add_subject import build_task as build_task2_add_subject
from adaptive_learning_system.tasks.task3_triage import build_task as build_task3_triage

from .attention import build_attention_summary
from .belief import apply_decay as apply_belief_decay, update_belief
from .concept_graph import ConceptGraph
from .hidden_state import apply_decay as apply_hidden_decay, apply_learning_event, mastery_to_stage
from .observation import generate_observation
from .reward import compute_reward
from .types import (
    BeliefConceptState,
    HiddenConceptState,
    ScenarioDefinition,
    StudentObservation,
    TopicSignal,
)

SCENARIOS = {
    "task1_review": build_task1_review,
    "task2_add_subject": build_task2_add_subject,
    "task3_triage": build_task3_triage,
}

TASK_GRADERS = {
    "task1_review": grade_task1_review,
    "task2_add_subject": grade_task2_add_subject,
    "task3_triage": grade_task3_triage,
}


class AdaptiveLearningRuntime:
    """Stateful runtime that owns all hidden state and belief tracking."""

    def __init__(self, task_name: str | None = None, seed: int = 7):
        self._configured_task = task_name or os.getenv("ADAPTIVE_LEARNING_TASK", DEFAULT_TASK)
        self._seed = seed
        self._rng = random.Random(seed)
        self._policy = AdaptiveLearningPolicyAgent()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: ScenarioDefinition | None = None
        self._graph = ConceptGraph()
        self._hidden_state: dict[str, HiddenConceptState] = {}
        self._belief_state: dict[str, BeliefConceptState] = {}
        self._session_elapsed_seconds = 0.0
        self._last_observation: StudentObservation | None = None
        self._last_reward = 0.0
        self._last_notes: list[str] = []
        self._last_action_error: str | None = None
        self._available_topic_signals: list[TopicSignal] = []

    @property
    def state(self) -> State:
        return self._state

    def reset(self, task_name: str | None = None) -> AdaptiveLearningSystemObservation:
        """Load the configured scenario and return the first summary."""

        if task_name:
            self._configured_task = task_name

        scenario_key = self._configured_task if self._configured_task in SCENARIOS else DEFAULT_TASK
        # Re-seed every episode so a fixed seed reproduces the same rollout after reset.
        self._rng = random.Random(self._seed)
        self._scenario = SCENARIOS[scenario_key]()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._graph = ConceptGraph(self._scenario.concept_nodes)
        self._hidden_state = copy.deepcopy(self._scenario.hidden_state)
        self._belief_state = copy.deepcopy(self._scenario.belief_state)
        self._session_elapsed_seconds = 0.0
        self._last_observation = None
        self._last_reward = 0.0
        self._last_notes = [self._scenario.description]
        self._last_action_error = None
        self._available_topic_signals = list(self._scenario.pending_topics)
        self._refresh_topic_signals()
        return self._build_summary("Reset")

    def step(self, action: AdaptiveLearningSystemAction) -> AdaptiveLearningSystemObservation:
        """Advance the scenario by one turn."""

        if self._scenario is None:
            return self.reset()

        previous_mastery = {
            concept_id: belief.estimated_mastery
            for concept_id, belief in self._belief_state.items()
        }
        self._advance_time(STEP_SECONDS)

        available_actions = self._available_actions()
        chosen_action = self._normalize_action(action, available_actions)
        invalid_action = self._action_signature(action) != self._action_signature(chosen_action)
        self._last_action_error = (
            f"invalid_action:{self._action_signature(action)}"
            if invalid_action
            else None
        )
        observation = self._execute_action(chosen_action)

        reward, notes = compute_reward(
            previous_beliefs=previous_mastery,
            beliefs=self._belief_state,
            hidden_state=self._hidden_state,
            observation=observation,
            action_type=chosen_action.action_type,
            invalid_action=invalid_action,
        )
        self._last_reward = reward
        self._last_notes = notes or ["No major belief shift this turn."]
        self._state.step_count += 1
        self._refresh_topic_signals()

        summary = self._build_summary(self._action_signature(chosen_action))
        summary.reward = reward
        summary.done = self._state.step_count >= self._scenario.step_budget
        return summary

    def _advance_time(self, elapsed_seconds: float) -> None:
        decay_rates = {
            concept_id: belief.decay_rate
            for concept_id, belief in self._belief_state.items()
        }
        apply_hidden_decay(self._hidden_state, decay_rates, elapsed_seconds)
        apply_belief_decay(self._belief_state, elapsed_seconds)
        self._session_elapsed_seconds += elapsed_seconds

    def _refresh_topic_signals(self) -> None:
        if not self._scenario:
            return
        unlocked = self._scenario.topic_unlock_schedule.get(self._state.step_count, [])
        existing = {signal.topic_name for signal in self._available_topic_signals}
        for signal in unlocked:
            if signal.topic_name not in existing:
                self._available_topic_signals.append(signal)

    def _normalize_action(
        self,
        action: AdaptiveLearningSystemAction,
        available_actions: list[str],
    ) -> AdaptiveLearningSystemAction:
        if self._action_signature(action) in available_actions:
            return action
        return AdaptiveLearningSystemAction(action_type="Wait")

    def _action_signature(self, action: AdaptiveLearningSystemAction) -> str:
        if action.action_type in {"Review", "Introduce", "Assess"} and action.concept_id:
            return f"{action.action_type}({action.concept_id})"
        if action.action_type == "AddSubject" and action.topic_name:
            return f"AddSubject({action.topic_name})"
        return "Wait"

    def _available_actions(self) -> list[str]:
        review_actions = [
            f"Review({node.concept_id})"
            for node in self._graph.all_nodes()
            if self._belief_state[node.concept_id].estimated_mastery > 0.0
        ]
        assess_actions = [
            f"Assess({node.concept_id})"
            for node in self._graph.all_nodes()
            if self._belief_state[node.concept_id].estimated_mastery >= LEARNED_THRESHOLD
        ]
        introduce_actions = [
            f"Introduce({node.concept_id})"
            for node in self._graph.ready_to_introduce(self._belief_state, LEARNED_THRESHOLD)
        ]
        add_subject_actions = [
            f"AddSubject({signal.topic_name})"
            for signal in self._available_topic_signals
        ]
        return sorted(
            {
                "Wait",
                *review_actions,
                *assess_actions,
                *introduce_actions,
                *add_subject_actions,
            }
        )

    def _execute_action(
        self,
        action: AdaptiveLearningSystemAction,
    ) -> StudentObservation | None:
        if action.action_type == "Wait":
            self._last_observation = None
            return None

        if action.action_type == "AddSubject" and action.topic_name:
            signal = next(
                (item for item in self._available_topic_signals if item.topic_name == action.topic_name),
                None,
            )
            if signal:
                self._merge_topic(signal)
                self._available_topic_signals = [
                    item for item in self._available_topic_signals if item.topic_name != action.topic_name
                ]
                self._last_notes = [signal.reason]
            else:
                self._last_action_error = f"unknown_topic:{action.topic_name}"
            self._last_observation = None
            return None

        concept_id = action.concept_id
        if not concept_id:
            self._last_observation = None
            return None

        if action.action_type == "Introduce":
            self._belief_state[concept_id].estimated_mastery = max(
                self._belief_state[concept_id].estimated_mastery,
                0.18,
            )

        observation = generate_observation(
            self._rng,
            concept_id,
            self._hidden_state[concept_id],
            action.action_type,
        )
        apply_learning_event(self._hidden_state[concept_id], observation.outcome, action.action_type)
        self._last_observation = update_belief(
            self._belief_state[concept_id],
            observation,
            self._session_elapsed_seconds,
        )
        return self._last_observation

    def _merge_topic(self, signal: TopicSignal) -> None:
        if not self._scenario:
            return

        added_ids = self._graph.merge_topic(self._scenario.topic_registry.get(signal.topic_name, []))
        for concept_id in added_ids:
            node = self._graph.get(concept_id)
            if signal.assessment_score < 0.3:
                initial_mastery = 0.0
            elif signal.assessment_score < 0.7:
                initial_mastery = 0.35 if node.difficulty <= 0.35 else 0.0
            else:
                initial_mastery = 0.62 if node.difficulty <= 0.35 else 0.18

            self._hidden_state[concept_id] = HiddenConceptState(
                true_mastery=initial_mastery,
                time_since_last_seconds=0.0,
                exposure_count=0,
                stage=mastery_to_stage(initial_mastery),
                consecutive_incorrect=0,
            )
            self._belief_state[concept_id] = BeliefConceptState(
                estimated_mastery=initial_mastery,
                confidence=0.20,
                decay_rate=BASE_DECAY_RATE,
                observation_count=0,
                last_observed_at_seconds=None,
            )

    def _graph_context(self) -> str:
        blocked = self._graph.blocked_concepts(self._belief_state, LEARNED_THRESHOLD)
        unlockable = [signal.topic_name for signal in self._available_topic_signals]
        if unlockable:
            return f"New topics ready: {', '.join(sorted(unlockable))}."
        if blocked:
            return f"Blocked by prerequisites: {', '.join(blocked[:3])}."
        return "Frontier is open; maintenance dominates exploration."

    def _build_summary(self, chosen_action: str) -> AdaptiveLearningSystemObservation:
        snapshot, aggregate = build_attention_summary(
            self._graph,
            self._belief_state,
            self._hidden_state,
            self._session_elapsed_seconds,
            self._last_observation,
            [signal.topic_name for signal in self._available_topic_signals],
        )
        grader_score, task_metrics = self._current_score()
        summary = AdaptiveLearningSystemObservation(
            task_name=self._scenario.name if self._scenario else DEFAULT_TASK,
            summary_message=self._scenario.description if self._scenario else "",
            chosen_action=chosen_action,
            belief_snapshot=snapshot,
            aggregate_stats=aggregate,
            last_observation=(
                ObservationDetails(
                    concept_id=self._last_observation.concept_id,
                    outcome=self._last_observation.outcome,
                    response_time_seconds=self._last_observation.response_time_seconds,
                    hint_used=self._last_observation.hint_used,
                    predicted_correct=self._last_observation.predicted_correct,
                )
                if self._last_observation
                else None
            ),
            available_actions=self._available_actions(),
            pending_topics=[
                PendingTopicSummary(
                    topic_name=signal.topic_name,
                    assessment_score=signal.assessment_score,
                    reason=signal.reason,
                )
                for signal in sorted(
                    self._available_topic_signals,
                    key=lambda item: (item.topic_name, item.assessment_score),
                )
            ],
            graph_context=self._graph_context(),
            recommended_action="",
            notes=self._last_notes,
            last_action_error=self._last_action_error,
            reward=self._last_reward,
            grader_score=grader_score,
            task_metrics=task_metrics,
            done=False,
        )
        summary.recommended_action = self._policy.choose_action(summary)
        return summary

    def _current_score(self) -> tuple[float, dict[str, float]]:
        task_name = self._scenario.name if self._scenario else DEFAULT_TASK
        grader = TASK_GRADERS.get(task_name)
        if grader is None or self._scenario is None:
            return 0.0, {}
        return grader(self._belief_state, self._hidden_state, self._graph, self._scenario)
