# Adaptive Learning System

Adaptive Learning System is an OpenEnv benchmark environment for tutoring decisions under uncertainty. The agent does not directly observe true student mastery. Instead, it receives a compact belief-state summary and must choose among constrained tutoring actions such as `Review`, `Assess`, `Introduce`, `AddSubject`, and `Wait`.

This repository is structured for the Scaler OpenEnv hackathon submission format:

- `adaptive_learning_system/`: the environment package and deployment target
- `adaptive_learning_system/inference.py`: submission inference runner with the required `[START]`, `[STEP]`, `[END]` log format
- `adaptive_learning_system/inference_local.py`: local helper for OpenAI-compatible local or cloud endpoints such as Ollama-style APIs
- `adaptive_learning_system/validate_submission.py`: local validator for OpenEnv, Docker, and inference log compliance
- `hackathon/`: reference material, sample scripts, and the challenge brief

## Why This Environment

We chose adaptive tutoring because it is a realistic sequential decision problem with partial observability:

- the system cannot see true mastery directly
- student performance is noisy
- knowledge decays over time
- new curriculum should be introduced only when prerequisites are stable
- every tutoring action has opportunity cost because step budgets are short

That makes the environment a natural fit for OpenEnv and aligns well with the hackathon emphasis on real-world utility, meaningful graders, and reward functions that reflect partial progress rather than only binary success.

## Environment Design

The runtime models a student as a hidden state plus a public belief state.

- Hidden state tracks true mastery, time since last practice, exposure count, and repeated incorrect streaks.
- Belief state tracks estimated mastery, confidence, adaptive decay rate, and last observation time.
- Observations expose only a compact attentional snapshot, aggregate statistics, valid actions, graph context, live task score, and a deterministic fallback action.
- Actions are filtered by the environment so the policy must operate inside realistic constraints such as prerequisites and unlocked subjects.

The environment is deterministic for a fixed seed and fixed action history. We explicitly re-seed on `reset()`, so identical rollouts reproduce the same prompts, observations, rewards, and task metrics.

## Concept Graph And Dynamic Subject Expansion

The concept graph is a real part of the environment, not just a description in the README. It follows the architecture in `hackathon/architecture.pdf`:

- the curriculum is a directed acyclic graph of concepts
- each concept node carries `topic`, `difficulty`, `content_ref`, and `prerequisites`
- introduction is prerequisite-gated
- new topics are merged at runtime through `AddSubject(topic_name)`

In the current implementation:

- `adaptive_learning_system/env/concept_graph.py` stores the DAG as nodes with prerequisite tuples rather than a separate edge table
- `prerequisites_met(...)` checks whether every prerequisite concept is at or above the learned threshold
- `ready_to_introduce(...)` exposes only unseen concepts whose prerequisites are satisfied
- `merge_topic(...)` grafts a new topic subgraph into the existing graph without duplicating shared nodes

That means the graph is doing real work in the environment:

- task 1 exposes `Introduce(quadratics)` only because `linear_equations` already meets the prerequisite threshold
- task 2 unlocks calculus as a new topic and merges `rate_of_change`, `limits`, and `derivatives` into the active graph
- after merging, downstream concepts still respect dependency checks, so advanced concepts do not become fully ready just because the topic was added

One honest limitation: the graph is operationally correct for filtering and topic merging, but the observation-side `graph_context` is still a compact summary rather than a rich dependency explanation. So the graph logic is there, but the agent sees only the most decision-relevant slice of it.

## Task Progression

The benchmark contains three tasks that intentionally increase in difficulty.

### Task 1: `task1_review`

Goal: preserve algebra mastery with a limited review budget.

Why it is the easiest:

- only one active curriculum is present
- no new subject must be integrated
- the problem is mostly maintenance and prioritization

What the agent must learn:

- identify weak or decaying concepts
- spend turns on the concepts most likely to fall below retention
- avoid wasting steps on no-op behavior

### Task 2: `task2_add_subject`

Goal: introduce calculus without sacrificing the original algebra concepts.

Why it is harder than task 1:

- the agent inherits the maintenance problem from task 1
- it must also react to a new topic unlock event
- adding the new subject too early can collapse the old curriculum
- the new subject is a grafted concept subgraph, so not all expansion is equally valuable

What the agent must learn:

- stabilize existing mastery first
- decide when subject expansion is safe
- build useful calculus foundations in graph order rather than merely unlocking the topic

### Task 3: `task3_triage`

Goal: rescue the initially struggling concepts under a shorter step budget.

Why it is the hardest:

- more concepts are simultaneously at risk
- the horizon is shorter
- several concepts begin already below threshold or on incorrect streaks
- strong concepts can distract the policy, but triage should focus on the fragile set

What the agent must learn:

- identify which concepts are truly in crisis
- rescue weak concepts instead of padding safe ones
- trade off immediate stabilization against broader session coverage

## Grader Rationale

The graders are designed to be causally linked to each task objective rather than just scoring a generic final state.

### Task 1 Grader

Task 1 uses:

- `average_mastery`
- `retained_ratio`

Reasoning:

- the task is fundamentally about preservation across the whole working curriculum
- average mastery captures broad quality
- retained ratio ensures the score reflects threshold-sensitive maintenance, not just a few strong concepts

### Task 2 Grader

Task 2 uses:

- `protected_retained_ratio`
- `protected_preservation_ratio`
- `collapse_ratio`
- `introduced_ratio`
- `rate_of_change_progress`
- `limits_progress`
- `derivatives_progress`

Reasoning:

- a good solution must protect the starting algebra curriculum, so old concepts are explicitly measured against the initial scenario
- `collapse_ratio` penalizes expansion that destroys previously stable concepts
- `introduced_ratio` ensures the score stays low if the agent never actually adds calculus
- the calculus progression metrics reward meaningful, prerequisite-aware subject integration rather than just unlocking new nodes into the graph

In short, task 2 only scores well when the agent expands safely and productively.

### Task 3 Grader

Task 3 uses:

- `focus_average_mastery`
- `focus_retained_ratio`
- `focus_supported_ratio`
- `rescue_gain_ratio`
- `unresolved_risk_ratio`
- `regression_ratio`

Reasoning:

- the grader focuses only on the concepts that were already vulnerable at the start of the scenario
- `rescue_gain_ratio` measures whether those weak concepts actually improved relative to the initial state
- `unresolved_risk_ratio` and `regression_ratio` prevent the agent from scoring decently while ignoring struggling concepts

This directly blocks the earlier exploit where repeatedly acting on a strong concept could still earn a nontrivial score.

## Reward Rationale

Rewards are shaped for learning signal, while final task quality comes from the graders.

The reward function is intentionally practical:

- productive turns gain reward from mastery improvements, threshold recovery, correct retention after long gaps, safe introductions, and stable subject expansion
- harmful turns lose reward through decay, threshold drops, incorrect or skipped responses, frustration buildup, invalid actions, and `Wait`
- `Wait` is near-zero and explicitly penalized by unresolved need, so no-op behavior does not look neutral

This gives the agent dense feedback while keeping the benchmark score anchored to task-specific outcomes.

## Reproducibility And Sanity Checks

We added regression tests around the exact failure modes that matter for a benchmark environment:

- reset determinism for fixed seeds and fixed action sequences
- `Wait` reward staying near zero
- recommended actions always being valid
- concept-graph prerequisite gating and runtime topic merging
- task 2 requiring real subject expansion
- task 3 penalizing the strong-concept spam exploit

Run them with:

```bash
python -m unittest adaptive_learning_system.tests.test_sanity -v
```

## Running Locally

### 1. Validate the environment

```bash
.\.venv\Scripts\openenv.exe validate .\adaptive_learning_system -v
```

### 2. Start the server locally

```bash
python -m adaptive_learning_system.server.app --port 8000
```

### 3. Run local inference

For a local or OpenAI-compatible model endpoint:

```bash
python -m adaptive_learning_system.inference_local
```

For the submission-oriented script:

```bash
python -m adaptive_learning_system.inference
```

### 4. Run the submission validator

```bash
python -m adaptive_learning_system.validate_submission
```

## Submission Notes

`adaptive_learning_system/inference.py` follows the hackathon contract:

- uses the OpenAI client
- runs all benchmark tasks
- emits strict `[START]`, `[STEP]`, `[END]` logs
- surfaces `last_action_error`
- keeps scores and rewards in `[0, 1]`

The Docker/OpenEnv deployment target lives under `adaptive_learning_system/`, and the environment passes `openenv validate`.

## Project Layout

```text
.
|-- adaptive_learning_system/
|   |-- inference.py
|   |-- inference_local.py
|   `-- validate_submission.py
|-- hackathon/
`-- README.md
```

For package-level environment details, see `adaptive_learning_system/README.md`.
