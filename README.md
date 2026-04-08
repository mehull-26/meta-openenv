---
title: Adaptive Learning System Environment Server
emoji: robot
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - pomdp
  - education
---

# Adaptive Learning System

An OpenEnv environment for adaptive tutoring decisions under partial observability. The runtime owns hidden student mastery, belief updates, forgetting, action filtering, rewards, and task-specific grading. The exposed agent is intentionally thin so graders judge the environment dynamics rather than hardcoded task logic.

## What The Environment Models

- Hidden mastery for each concept, plus forgetting over time.
- Belief-state updates from noisy student observations.
- Action-constrained tutoring decisions: `Review`, `Assess`, `Introduce`, `AddSubject`, and `Wait`.
- Curriculum growth through prerequisite-aware subject expansion.
- Task-specific grading with normalized scores in `[0, 1]`.

## Tasks

- `task1_review`: preserve algebra mastery across a limited review budget.
- `task2_add_subject`: add calculus while protecting the original algebra concepts.
- `task3_triage`: rescue the initially struggling concepts under a short horizon.

## Observation Surface

Each `reset()` and `step()` returns a compact observation designed for LLM control:

- `belief_snapshot`: attentional concept window with mastery, urgency, confidence, and prerequisite status.
- `aggregate_stats`: concept counts, below-threshold count, average mastery, elapsed session time, unlockable topics.
- `last_observation`: noisy student response from the previous interaction.
- `available_actions`: valid actions after prerequisite and topic filtering.
- `graph_context`: short curriculum or unlock-status summary.
- `recommended_action`: deterministic valid fallback from the built-in heuristic.
- `grader_score` and `task_metrics`: live task-specific scoring signals.
- `last_action_error`: error text when an invalid action is filtered or a topic request is unknown.

## Reward Design

Rewards are bounded to `[0, 1]` and are intentionally near zero for unproductive turns.

- Positive utility comes from mastery gains, recovering concepts above the retention threshold, correct or partial responses, long-gap retention wins, safe introductions, and stable subject expansion.
- Negative utility comes from mastery loss, threshold drops, incorrect or skipped responses, repeated frustration, invalid actions, and `Wait`.
- `Wait` is explicitly penalized by unresolved learning need, so no-op play does not appear neutral.

The reward is a shaped learning signal. Final benchmark quality is still determined by the task graders.

## Grader Design

- `task1_review`: scores average mastery plus retained ratio across the active curriculum.
- `task2_add_subject`: scores actual calculus introduction and prerequisite-aware progress, but discounts heavily when the original algebra set collapses.
- `task3_triage`: scores rescue of the initially vulnerable concepts, not generic calmness on already-strong concepts.

## Determinism

The environment is deterministic for a fixed seed and a fixed action history.

- Reset re-seeds the runtime RNG, so the same task plus the same action sequence reproduces the same observations, rewards, and grader metrics.
- LLM-driven inference can still vary because the model may choose different actions, but identical environment histories produce identical model prompts.

## OpenEnv Workflow

```bash
# From this directory
..\.venv\Scripts\openenv.exe validate . -v

# Run the local server
python -m adaptive_learning_system.server.app --port 8000

# Build the Docker image used by OpenEnv / HF Spaces
docker build -t adaptive_learning_system-env:latest .
```

If you want to start directly on a different scenario:

```powershell
$env:ADAPTIVE_LEARNING_TASK = "task2_add_subject"
python -m adaptive_learning_system.server.app
```

## Deploy Only This Directory To Hugging Face Spaces

This folder is already structured so it can be the root of a Docker Space by itself.

- `README.md` contains the Hugging Face YAML front matter at the top.
- `Dockerfile` starts the FastAPI server for the environment.
- `requirements.txt` contains the runtime dependencies needed by the container.

That means you should push the contents of `adaptive_learning_system/` to the Space repo root, not the repository root above it.

### Recommended Flow

1. Create a new Hugging Face Space with `SDK = Docker`.
2. Open a terminal in `adaptive_learning_system/`.
3. Run the helper script:

```powershell
.\scripts\push_to_hf_space.ps1 -SpaceId your-hf-username/your-space-name
```

The script will:

- initialize a git repo in this folder if needed
- add or update a `space` remote pointing at `https://huggingface.co/spaces/<SPACE_ID>`
- commit the current folder contents
- push only this directory to the Space

### Manual Git Alternative

If you prefer to do it yourself:

```bash
cd adaptive_learning_system
git init
git remote add space https://huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>
git add .
git commit -m "Initial HF Spaces deployment"
git push --set-upstream space HEAD:main
```

### Notes

- The Space metadata lives in this folder's `README.md`, which matches Hugging Face's requirement that the YAML block be at the root of the Space repository.
- This Docker Space is configured to run on port `8000`, and the README metadata sets `app_port: 8000` to match.
- Root-level files such as `..\inference.py` are intentionally excluded from this deployment path.

## Submission Scripts

The hackathon-facing scripts live in the repository root:

- `..\inference.py`: submission inference runner using an OpenAI-compatible API.
- `..\inference_local.py`: local helper for Ollama-style or other OpenAI-compatible endpoints.
- `..\validate_submission.py`: local validation helper for deployability and log-format checks.

## Sanity Tests

A lightweight sanity suite lives in `tests/test_sanity.py`. It checks:

- reset determinism for a fixed seed and fixed action sequence
- `Wait` reward staying near zero
- recommended actions being valid at reset
- task 2 grading requiring actual subject expansion
- task 3 grading penalizing the old strong-concept spam exploit

Run it from the repository root with:

```bash
python -m unittest adaptive_learning_system.tests.test_sanity -v
```

## Project Structure

```text
adaptive_learning_system/
|-- agent/
|   `-- agent.py
|-- config/
|   `-- defaults.py
|-- env/
|   |-- attention.py
|   |-- belief.py
|   |-- concept_graph.py
|   |-- environment.py
|   |-- hidden_state.py
|   |-- observation.py
|   `-- reward.py
|-- server/
|   |-- adaptive_learning_system_environment.py
|   `-- app.py
|-- tasks/
|   |-- grading.py
|   |-- task1_review.py
|   |-- task2_add_subject.py
|   `-- task3_triage.py
|-- tests/
|   `-- test_sanity.py
|-- client.py
|-- models.py
|-- openenv.yaml
`-- pyproject.toml
```
