"""OpenEnv wrapper around the adaptive learning runtime."""

from __future__ import annotations

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..env.environment import AdaptiveLearningRuntime
    from ..models import AdaptiveLearningSystemAction, AdaptiveLearningSystemObservation
except ImportError:
    from adaptive_learning_system.env.environment import AdaptiveLearningRuntime
    from adaptive_learning_system.models import (
        AdaptiveLearningSystemAction,
        AdaptiveLearningSystemObservation,
    )


class AdaptiveLearningSystemEnvironment(Environment):
    """OpenEnv-compatible wrapper that exposes the runtime over HTTP/WebSocket."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._runtime = AdaptiveLearningRuntime()

    def reset(self, task_name: str | None = None) -> AdaptiveLearningSystemObservation:
        return self._runtime.reset(task_name=task_name)

    def step(self, action: AdaptiveLearningSystemAction) -> AdaptiveLearningSystemObservation:  # type: ignore[override]
        return self._runtime.step(action)

    @property
    def state(self) -> State:
        return self._runtime.state
