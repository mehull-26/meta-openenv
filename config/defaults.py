"""Static configuration for the POMDP adaptive learning environment."""

from __future__ import annotations

DEFAULT_TASK = "task1_review"

STEP_SECONDS = 6 * 60 * 60
RETENTION_THRESHOLD = 0.55
MASTERED_THRESHOLD = 0.85
LEARNED_THRESHOLD = 0.50

BASE_DECAY_RATE = 0.22
BASE_ADAPTATION_RATE = 0.10
ADAPTATION_ALPHA = 0.30
BASE_LEARNING_BOOST = 0.24
INCORRECT_PENALTY_FACTOR = 0.70
INTRODUCTION_BASELINE = 0.28
ASSESSMENT_READY_THRESHOLD = 0.70

ATTENTION_SLOTS = {
    "critical": 3,
    "frontier": 3,
    "ready": 2,
    "recent": 1,
}

REWARD_WEIGHTS = {
    "mastery_gain": 3.5,
    "mastery_loss": 1.25,
    "threshold_recovery": 0.9,
    "threshold_drop": 1.4,
    "correct_bonus": 0.45,
    "partial_bonus": 0.20,
    "incorrect_penalty": 0.30,
    "skipped_penalty": 0.40,
    "retention_bonus": 0.30,
    "frustration_penalty": 0.90,
    "exploration_bonus": 0.20,
    "subject_expansion_bonus": 0.35,
    "wait_penalty": 0.35,
    "unmet_need_penalty": 0.70,
    "invalid_action_penalty": 0.80,
    "reward_floor": 0.02,
}

TOPIC_LIBRARY = {
    "algebra": [
        {
            "concept_id": "number_sense",
            "difficulty": 0.10,
            "content_ref": "algebra/number-sense",
            "prerequisites": [],
        },
        {
            "concept_id": "fractions",
            "difficulty": 0.20,
            "content_ref": "algebra/fractions",
            "prerequisites": ["number_sense"],
        },
        {
            "concept_id": "linear_equations",
            "difficulty": 0.35,
            "content_ref": "algebra/linear-equations",
            "prerequisites": ["fractions"],
        },
        {
            "concept_id": "inequalities",
            "difficulty": 0.45,
            "content_ref": "algebra/inequalities",
            "prerequisites": ["linear_equations"],
        },
        {
            "concept_id": "quadratics",
            "difficulty": 0.65,
            "content_ref": "algebra/quadratics",
            "prerequisites": ["linear_equations"],
        },
        {
            "concept_id": "graph_reading",
            "difficulty": 0.30,
            "content_ref": "algebra/graph-reading",
            "prerequisites": ["linear_equations"],
        },
    ],
    "calculus": [
        {
            "concept_id": "rate_of_change",
            "difficulty": 0.30,
            "content_ref": "calculus/rate-of-change",
            "prerequisites": ["graph_reading"],
        },
        {
            "concept_id": "limits",
            "difficulty": 0.55,
            "content_ref": "calculus/limits",
            "prerequisites": ["rate_of_change"],
        },
        {
            "concept_id": "derivatives",
            "difficulty": 0.70,
            "content_ref": "calculus/derivatives",
            "prerequisites": ["limits", "quadratics"],
        },
    ],
    "geometry": [
        {
            "concept_id": "angle_basics",
            "difficulty": 0.15,
            "content_ref": "geometry/angle-basics",
            "prerequisites": ["number_sense"],
        },
        {
            "concept_id": "triangles",
            "difficulty": 0.35,
            "content_ref": "geometry/triangles",
            "prerequisites": ["angle_basics"],
        },
        {
            "concept_id": "coordinate_geometry",
            "difficulty": 0.60,
            "content_ref": "geometry/coordinate-geometry",
            "prerequisites": ["graph_reading", "triangles"],
        },
    ],
}

TASK_REGISTRY = {
    "task1_review": {
        "description": "Core review scheduling with a small algebra curriculum.",
        "step_budget": 16,
    },
    "task2_add_subject": {
        "description": "Introduce calculus without letting existing algebra mastery collapse.",
        "step_budget": 18,
    },
    "task3_triage": {
        "description": "Stabilize multiple decaying concepts under a short step budget.",
        "step_budget": 12,
    },
}
