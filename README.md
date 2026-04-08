---
title: Adaptive Learning System Environment Server
emoji: "🤖"
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - pomdp
  - education
---

# Adaptive Learning System

An OpenEnv benchmark environment for adaptive tutoring under partial observability. The environment owns hidden student mastery, belief updates, forgetting, prerequisite gating, rewards, and task-specific grading. The agent-facing surface is intentionally compact so evaluation focuses on environment quality rather than hand-coded policy logic.

## Submission Surface

This directory is intentionally self-contained for OpenEnv and Docker submission:

- `env/`, `tasks/`, `models.py`, and `server/`: the runtime and API surface
- `inference.py`: the submission runner that emits `[START]`, `[STEP]`, and `[END]` logs
- `openenv.yaml`, `Dockerfile`, `requirements.txt`, and `pyproject.toml`: deployment and packaging metadata
- `validate_submission.py`: pre-submission validation for OpenEnv, Docker, and inference log checks
- `tests/test_sanity.py`: small regression coverage for deterministic benchmark behavior

Duplicate docs and deployment convenience scripts were removed, while `validate_submission.py` is retained as a submission check.

## Tasks

- `task1_review`: preserve algebra mastery across a limited review budget
- `task2_add_subject`: add calculus while protecting the original algebra concepts
- `task3_triage`: rescue the initially struggling concepts under a short horizon

## Observation And Action Model

Each step exposes only the decision-relevant slice of state:

- `belief_snapshot`: compact concept summaries with mastery, urgency, confidence, and prerequisite status
- `aggregate_stats`: global learning-health metrics and unlockable topics
- `pending_topics`: signals when `AddSubject(topic_name)` becomes useful
- `available_actions`: valid actions only, after prerequisite and topic filtering
- `recommended_action`, `grader_score`, `task_metrics`, and `last_action_error`: built-in guidance and live task feedback

Supported actions are `Review`, `Assess`, `Introduce`, `AddSubject`, and `Wait`.

## Reward And Grading

Rewards are bounded to `[0, 1]` and stay near zero for unproductive turns. Positive signal comes from mastery improvement, threshold recovery, safe expansion, and retention wins. Negative signal comes from decay, incorrect responses, repeated frustration, invalid actions, and unhelpful waiting.

Final quality is determined by task graders:

- task 1 measures retained algebra mastery
- task 2 rewards safe calculus expansion without collapsing the protected baseline
- task 3 rewards rescuing vulnerable concepts instead of over-serving already strong ones

The environment is deterministic for a fixed seed and action history.

## Local Checks

```bash
# Validate the environment contract from this directory
.\.venv\Scripts\python.exe -m openenv.cli validate . -v

# Run the local server from this directory
python -m server.app --port 7860

# Run the submission inference script from this directory
python inference.py

# Optional regression suite from the repository root above this directory
python -m unittest adaptive_learning_system.tests.test_sanity -v
```

If you want to start directly on a different task:

```powershell
$env:ADAPTIVE_LEARNING_TASK = "task2_add_subject"
python -m server.app --port 7860
```

## Deployment Note

This folder can be pushed directly as the root of a Docker Space. `README.md`, `Dockerfile`, `requirements.txt`, and `openenv.yaml` are already aligned with that layout.

## Project Structure

```text
adaptive_learning_system/
|-- agent/
|-- config/
|-- env/
|-- server/
|-- tasks/
|-- tests/
|-- client.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- validate_submission.py
`-- uv.lock
```
