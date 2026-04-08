"""Concept graph utilities for prerequisite-aware action filtering."""

from __future__ import annotations

from dataclasses import replace

from .types import BeliefConceptState, ConceptNode


class ConceptGraph:
    """Small DAG wrapper used by the environment."""

    def __init__(self, nodes: list[ConceptNode] | None = None):
        self._nodes: dict[str, ConceptNode] = {}
        if nodes:
            self.merge_topic(nodes)

    def merge_topic(self, nodes: list[ConceptNode]) -> list[str]:
        added: list[str] = []
        for node in nodes:
            if node.concept_id in self._nodes:
                continue
            self._nodes[node.concept_id] = replace(node)
            added.append(node.concept_id)
        return added

    def get(self, concept_id: str) -> ConceptNode:
        return self._nodes[concept_id]

    def all_nodes(self) -> list[ConceptNode]:
        return list(self._nodes.values())

    def prerequisites_met(
        self,
        concept_id: str,
        beliefs: dict[str, BeliefConceptState],
        threshold: float,
    ) -> bool:
        node = self.get(concept_id)
        return all(
            beliefs.get(prereq) is not None
            and beliefs[prereq].estimated_mastery >= threshold
            for prereq in node.prerequisites
        )

    def ready_to_introduce(
        self,
        beliefs: dict[str, BeliefConceptState],
        threshold: float,
    ) -> list[ConceptNode]:
        ready: list[ConceptNode] = []
        for node in self.all_nodes():
            belief = beliefs.get(node.concept_id)
            if belief is None or belief.estimated_mastery > 0.0:
                continue
            if self.prerequisites_met(node.concept_id, beliefs, threshold):
                ready.append(node)
        return sorted(ready, key=lambda item: (item.difficulty, item.concept_id))

    def blocked_concepts(
        self,
        beliefs: dict[str, BeliefConceptState],
        threshold: float,
    ) -> list[str]:
        blocked: list[str] = []
        for node in self.all_nodes():
            belief = beliefs.get(node.concept_id)
            if belief is None or belief.estimated_mastery > 0.0:
                continue
            if not self.prerequisites_met(node.concept_id, beliefs, threshold):
                blocked.append(node.concept_id)
        return blocked
