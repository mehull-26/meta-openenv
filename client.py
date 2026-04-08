"""Client for the adaptive learning OpenEnv package."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AdaptiveLearningSystemAction, AdaptiveLearningSystemObservation


class AdaptiveLearningSystemEnv(
    EnvClient[AdaptiveLearningSystemAction, AdaptiveLearningSystemObservation, State]
):
    """Typed client for the adaptive learning environment."""

    def _step_payload(self, action: AdaptiveLearningSystemAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[AdaptiveLearningSystemObservation]:
        obs_payload = dict(payload.get("observation", {}))
        obs_payload["reward"] = payload.get("reward")
        obs_payload["done"] = payload.get("done", False)
        observation = AdaptiveLearningSystemObservation.model_validate(obs_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
