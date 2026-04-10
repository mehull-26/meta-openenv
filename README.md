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

An OpenEnv benchmark environment for adaptive tutoring under partial observability. The environment owns hidden student mastery, belief updates, forgetting, prerequisite gating, rewards, and task-specific grading. The agent-facing surface is intentionally compact so evaluation focuses on scheduling and prioritization decisions rather than hand-coded policy logic.

## How the Environment Works

The environment is a **POMDP** (Partially Observable Markov Decision Process). The agent never sees the student's true mastery directly — it only sees the system's *beliefs* about mastery, which are updated after each observation.

| State | Description |
|---|---|
| **Hidden state** | True mastery, time since last practice, exposure count, consecutive incorrect answers, learning stage |
| **Belief state** | Estimated mastery, confidence, decay rate, observation count — updated each step via Bayesian inference |
| **Observation** | Outcome (correct / partial / incorrect / skipped), response time, hint usage, gap since last seen |

Each step represents **6 hours** of real time (`STEP_SECONDS = 6 * 60 * 60`). Mastery decays between steps according to each concept's decay rate and how long it has gone unreviewed.

## Concept Graph and Prerequisites

Three topics are defined, each with an internal prerequisite chain:

```
algebra
  number_sense (difficulty 0.10)
    └─ fractions (0.20)
         └─ linear_equations (0.35)
              ├─ inequalities (0.45)
              ├─ quadratics (0.65)
              └─ graph_reading (0.30)

calculus  (unlocked dynamically in task2)
  graph_reading ─► rate_of_change (0.30)
                     └─ limits (0.55)
                          └─ derivatives (0.70) ◄─ quadratics

geometry
  number_sense ─► angle_basics (0.15)
                    └─ triangles (0.35)
  graph_reading + triangles ─► coordinate_geometry (0.60)
```

A concept cannot be `Introduce`d until all its prerequisites are above the `LEARNED_THRESHOLD` (0.50). The `available_actions` field in each observation already filters to valid choices only.

## Actions

| Action | Effect |
|---|---|
| `Review(concept_id)` | Schedules a practice session; most likely to improve mastery on a known concept |
| `Assess(concept_id)` | Tests without teaching; updates belief confidence more than mastery; useful for calibration |
| `Introduce(concept_id)` | Teaches a new (unseen) concept; sets baseline mastery around 0.28; requires prerequisites |
| `AddSubject(topic_name)` | Unlocks all concepts in a new topic and adds them to the curriculum; only available when a `TopicSignal` is pending |
| `Wait` | Advances time without any learning action; penalized when learning needs are unmet |

## Key Thresholds

| Constant | Value | Meaning |
|---|---|---|
| `RETENTION_THRESHOLD` | 0.55 | Minimum mastery to count a concept as retained; crossing this boundary triggers reward signals |
| `MASTERED_THRESHOLD` | 0.85 | Mastery level considered fully mastered |
| `LEARNED_THRESHOLD` | 0.50 | Minimum mastery in a prerequisite before its dependents can be introduced |
| `ASSESSMENT_READY_THRESHOLD` | 0.70 | Minimum mastery before the system considers a concept ready for assessment-only practice |
| `BASE_DECAY_RATE` | 0.22 | Default per-step forgetting rate; increases if the student has been struggling |

## Reward Signal

Rewards are bounded to `[0, 1]` via `risk_discount × (floor + (1−floor) × (1−exp(−positive_utility)))` where `floor = 0.02`. Near-zero rewards occur for unproductive turns; reaching 1.0 requires simultaneous mastery gains, correct responses, and no negative signals.

**Positive utility components:**

| Component | Weight | Trigger |
|---|---|---|
| `mastery_gain` | 3.5 × gain | Belief mastery increased (continuous) |
| `threshold_recovery` | 0.9 per concept | Concept crossed back above `RETENTION_THRESHOLD` |
| `correct_bonus` | 0.45 × (1 + gap_factor) | Correct response; scales up after a long gap |
| `retention_bonus` | 0.30 | Correct after ≥12 hours without practice |
| `partial_bonus` | 0.20 × (1 + 0.5 × gap_factor) | Partial response |
| `exploration_bonus` | 0.20 (correct) / 0.10 (partial) | Successful concept introduction via `Introduce` |
| `subject_expansion_bonus` | 0.175–0.35 | New concepts added via `AddSubject`; scales with prior curriculum stability |

**Negative utility components:**

| Component | Weight | Trigger |
|---|---|---|
| `mastery_loss` | 1.25 × loss | Belief mastery decreased from decay |
| `threshold_drop` | 1.4 per concept | Concept fell below `RETENTION_THRESHOLD` |
| `incorrect_penalty` | 0.30 × (1 + 0.5 × gap_factor) | Incorrect response |
| `skipped_penalty` | 0.40 × (1 + 0.5 × gap_factor) | Skipped response (higher than incorrect: zero engagement) |
| `frustration_penalty` | 0.90 | ≥3 consecutive incorrect answers on one concept |
| `wait_penalty` | 0.35 + 0.70 × below_threshold_ratio | `Wait` action; scales with fraction of curriculum below threshold |
| `invalid_action_penalty` | 0.80 | Invalid action (filtered to `Wait`; stacks with `wait_penalty`) |

The gap factor is `clamp(gap_seconds / (18 × 3600))`, reaching 1.0 after 18 hours without practice. Correct responses after long gaps receive full gap scaling on the bonus; incorrect/partial/skipped receive half scaling.

## Tasks

### task1_review — Core Review Scheduling
**Step budget**: 16 steps (~4 days)  
**Curriculum**: 6 algebra concepts  
**Challenge**: Several concepts start below or near `RETENTION_THRESHOLD`. `quadratics` is unseen (mastery 0.00). The agent must prioritize review to prevent decay while optionally introducing `quadratics`.

| Concept | Estimated Mastery | True Mastery | Status |
|---|---|---|---|
| number_sense | 0.88 | 0.90 | stable |
| fractions | 0.64 | 0.68 | solid |
| linear_equations | 0.55 | 0.52 | at threshold |
| inequalities | 0.29 | 0.32 | struggling |
| quadratics | 0.00 | 0.00 | unseen |
| graph_reading | 0.42 | 0.47 | below threshold |

**Grader** (`grade_task1_review`): `0.7 × average_mastery + 0.3 × retained_ratio` across all concepts. Maximized by lifting all concepts above `RETENTION_THRESHOLD` while preventing decay.

---

### task2_add_subject — Curriculum Expansion
**Step budget**: 18 steps (~4.5 days)  
**Curriculum**: 6 algebra concepts (strong baseline) + calculus unlocks at step 4  
**Challenge**: The agent must stabilize algebra first, then expand into calculus without letting any initially-stable algebra concepts collapse below `RETENTION_THRESHOLD`.

Calculus unlock at step 4 carries `assessment_score=0.78`, signaling the student is ready. Calculus prerequisites chain: `rate_of_change → limits → derivatives` (derivatives also requires `quadratics`).

**Grader** (`grade_task2_add_subject`):
- `maintenance_score = 0.60 × protected_retained_ratio + 0.40 × protected_preservation_ratio`
- `expansion_score = 0.30 × introduced_ratio + 0.45 × mean(rate_of_change_progress, limits_progress) + 0.25 × derivatives_progress`
- `score = introduced_ratio × (0.60 × maintenance + 0.40 × expansion) × (1 − 0.75 × collapse_ratio)`

`introduced_ratio` gates the entire score — failing to introduce any calculus concepts yields 0 regardless of algebra maintenance. `collapse_ratio` penalizes up to 75% if initially-stable concepts fall below threshold.

---

### task3_triage — High-Pressure Rescue
**Step budget**: 12 steps (~3 days)  
**Curriculum**: 6 algebra + 3 geometry = 9 concepts  
**Challenge**: Multiple concepts are below threshold, two with consecutive incorrect answers already logged. The agent must triage and rescue the weakest concepts without triggering frustration or letting already-fragile ones regress.

| Concept | Estimated Mastery | True Mastery | Consecutive Incorrect |
|---|---|---|---|
| number_sense | 0.85 | 0.88 | 0 (not a focus) |
| fractions | 0.47 | 0.44 | 1 |
| linear_equations | 0.42 | 0.37 | 2 (one more triggers frustration) |
| inequalities | 0.31 | 0.28 | 0 |
| quadratics | 0.24 | 0.21 | 0 |
| graph_reading | 0.45 | 0.41 | 0 |
| angle_basics | 0.52 | 0.49 | 0 |
| triangles | 0.33 | 0.31 | 0 |
| coordinate_geometry | 0.00 | 0.00 | 0 (distractor — not tracked) |

**Grader** (`grade_task3_triage`) operates only on the 7 *focus concepts* (all with `0 < true_mastery < RETENTION_THRESHOLD` or `consecutive_incorrect > 0` at scenario start):
- `base_score = 0.35 × focus_average_mastery + 0.25 × focus_supported_ratio + 0.20 × focus_retained_ratio + 0.20 × rescue_gain_ratio`
- `risk_discount = (1 − 0.65 × unresolved_risk_ratio) × (1 − 0.50 × regression_ratio)`
- `score = base_score × max(0, risk_discount)`

`rescue_gain_ratio` is normalized to a target average gain of 0.18 across focus concepts (calibrated to the gap between starting average and retention threshold). `focus_supported_ratio` counts concepts at ≥0.40 mastery with fewer than 2 consecutive incorrect.

## Observation and Action Model

Each step exposes only the decision-relevant slice of state:

- `belief_snapshot`: compact concept summaries with mastery, urgency, confidence, and prerequisite status
- `aggregate_stats`: global learning-health metrics and unlockable topics
- `pending_topics`: signals when `AddSubject(topic_name)` becomes available
- `available_actions`: valid actions only, after prerequisite and topic filtering
- `recommended_action`, `grader_score`, `task_metrics`, and `last_action_error`: built-in guidance and live task feedback

## Submission Surface

This directory is self-contained for OpenEnv and Docker submission:

- `env/`, `tasks/`, `models.py`, and `server/`: the runtime and API surface
- `inference.py`: the submission runner that emits `[START]`, `[STEP]`, and `[END]` logs
- `openenv.yaml`, `Dockerfile`, `requirements.txt`, and `pyproject.toml`: deployment and packaging metadata
- `validate_submission.py`: pre-submission validation for OpenEnv, Docker, and inference log checks
- `tests/test_sanity.py`: small regression coverage for deterministic benchmark behavior

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

To start on a specific task:

```powershell
$env:ADAPTIVE_LEARNING_TASK = "task2_add_subject"
python -m server.app --port 7860
```

## Deployment Note

This folder can be pushed directly as the root of a Docker Space. `README.md`, `Dockerfile`, `requirements.txt`, and `openenv.yaml` are already aligned with that layout.

## Project Structure

```text
adaptive_learning_system/
|-- agent/          (reference agent)
|-- config/         (defaults.py: thresholds, weights, topic library, task registry)
|-- env/            (POMDP core: hidden state, beliefs, rewards, observations)
|-- server/         (FastAPI app and OpenEnv server adapter)
|-- tasks/          (scenario definitions and graders for each task)
|-- tests/          (sanity regression suite)
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
