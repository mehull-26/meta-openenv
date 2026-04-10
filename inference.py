"""Baseline inference runner for the Scaler OpenEnv submission contract."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from openai import OpenAI
from openenv.core.containers.runtime import LocalDockerProvider

from adaptive_learning_system import AdaptiveLearningSystemAction, AdaptiveLearningSystemEnv
from adaptive_learning_system.agent import choose_best_available_action


class _Port7860Provider(LocalDockerProvider):
    """LocalDockerProvider variant that maps to container port 7860 instead of 8000."""

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        if port is None:
            port = self._find_available_port()
        self._container_name = self._generate_container_name(image)
        cmd = ["docker", "run", "-d", "--name", self._container_name, "-p", f"{port}:7860"]
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.append(image)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self._container_id = result.stdout.strip()
        time.sleep(1)
        return f"http://localhost:{port}"

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or "missing-api-key"
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv(
    "IMAGE_NAME") or "adaptive_learning_system-env:latest"
_env_base_url = os.getenv("ENV_BASE_URL")
_space_url = os.getenv("SPACE_URL")
ENV_BASE_URL = (
    _env_base_url
    if _env_base_url is not None
    else _space_url
    if _space_url is not None
    else "https://mehull26-adaptive-learning-system.hf.space"
)
BENCHMARK = os.getenv("ADAPTIVE_LEARNING_BENCHMARK",
                      "adaptive_learning_system")
TEMPERATURE = float(os.getenv("INFERENCE_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("INFERENCE_MAX_TOKENS", "220"))
MAX_STEPS = int(os.getenv("INFERENCE_MAX_STEPS", "24"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.10"))

TASKS = [
    task.strip()
    for task in os.getenv(
        "ADAPTIVE_LEARNING_TASKS",
        "task1_review,task2_add_subject,task3_triage",
    ).split(",")
    if task.strip()
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are choosing actions for a POMDP-based adaptive learning environment.
    Your objective is to maximize final grader_score, not just immediate reward.
    Prioritize retention over exploration.
    Use only the provided available_actions.
    Avoid getting stuck repeating the same action when progress is weak.
    Prefer the weakest urgent concept, but switch concepts when the same review has
    already been attempted repeatedly or when another concept is similarly weak.
    Use Assess only when a concept appears ready, and use AddSubject only when the
    pending-topic signal is strong and existing skills look stable.
    Think briefly before answering.
    Reply with compact JSON only using this schema:
    {"action":"<exact action from available_actions>","reasoning":"<brief rationale>"}
    Keep reasoning concise.
    """
).strip()


def stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_observation(observation) -> str:
    snapshot_lines = [
        (
            f"- {item.concept_id} topic={item.topic} stage={item.stage} "
            f"mastery={item.estimated_mastery:.2f} confidence={item.confidence:.2f} "
            f"urgency={item.urgency:.2f} gap_h={item.time_since_review_hours:.1f} "
            f"streak={item.consecutive_incorrect}"
        )
        for item in observation.belief_snapshot
    ]
    pending_topic_lines = [
        (
            f"- {item.topic_name} assessment_score={item.assessment_score:.2f} "
            f"reason={item.reason}"
        )
        for item in observation.pending_topics
    ]
    last_observation = (
        "none"
        if observation.last_observation is None
        else (
            f"{observation.last_observation.concept_id} outcome={observation.last_observation.outcome} "
            f"response_time={observation.last_observation.response_time_seconds:.1f}s "
            f"hint_used={str(observation.last_observation.hint_used).lower()}"
        )
    )
    return textwrap.dedent(
        f"""
        Task: {observation.task_name}
        Goal: {observation.summary_message}
        Graph context: {observation.graph_context}
        Aggregate stats:
        - total_concepts={observation.aggregate_stats.total_concepts}
        - below_threshold={observation.aggregate_stats.below_threshold}
        - average_mastery={observation.aggregate_stats.average_mastery:.3f}
        - session_hours={observation.aggregate_stats.session_hours:.1f}
        - unlockable_topics={",".join(observation.aggregate_stats.unlockable_topics) or "none"}
        - concepts_with_incorrect_streak={observation.aggregate_stats.concepts_with_incorrect_streak}
        - frustration_risk_count={observation.aggregate_stats.frustration_risk_count}
        Last observation: {last_observation}
        Current grader_score: {observation.grader_score:.3f}
        Recommended action: {observation.recommended_action or "Wait"}
        Pending topics:
        {chr(10).join(pending_topic_lines) if pending_topic_lines else "- none"}
        Belief snapshot:
        {chr(10).join(snapshot_lines) if snapshot_lines else "- none"}
        Available actions:
        {chr(10).join(f"- {action}" for action in observation.available_actions)}
        """
    ).strip()


def build_policy_notes(observation, action_history: Iterable[str]) -> str:
    recent_entries = list(action_history)[-4:]
    recent_actions: list[str] = []
    for entry in recent_entries:
        match = re.search(r"action=(?P<action>\S+)", entry)
        if match:
            recent_actions.append(match.group("action"))

    notes: list[str] = []
    if recent_actions:
        repeated = recent_actions.count(recent_actions[-1])
        if repeated >= 2:
            notes.append(
                f"The recent policy is overusing {recent_actions[-1]}; only repeat it if it is still clearly the highest-value move."
            )

    if observation.task_name == "task1_review":
        notes.append(
            "Task 1: recover the weakest review concepts, but do not tunnel on one concept if another weak concept also needs attention."
        )
        notes.append(
            "When a concept seems stabilized, rotate to another weak concept or assess a concept that looks ready."
        )
    elif observation.task_name == "task2_add_subject":
        notes.append(
            "Task 2: preserve algebra first, then add the pending subject once current mastery is stable enough."
        )
    elif observation.task_name == "task3_triage":
        notes.append(
            "Task 3: rescue struggling concepts first; do not spend most turns on already strong concepts."
        )

    if observation.pending_topics:
        strongest_topic = max(
            observation.pending_topics, key=lambda item: item.assessment_score)
        notes.append(
            f"Strongest pending topic signal: {strongest_topic.topic_name} ({strongest_topic.assessment_score:.2f}); consider it only if retention looks stable."
        )

    if observation.aggregate_stats.frustration_risk_count > 0:
        notes.append(
            "There is active frustration risk, so prioritize support for weak or error-streak concepts over expansion."
        )

    if not notes:
        notes.append(
            "Choose the single best action for long-term score improvement, not the most repetitive action."
        )

    return "\n".join(f"- {note}" for note in notes)


def build_user_prompt(step: int, observation, action_history: Iterable[str]) -> str:
    action_history = list(action_history)
    history_text = "\n".join(action_history) if action_history else "None"
    return textwrap.dedent(
        f"""
        Step {step}
        {format_observation(observation)}

        Decision guidance:
        {build_policy_notes(observation, action_history)}

        Recent action history:
        {history_text}

        Reply as compact JSON only:
        {{"action":"<one exact action from available_actions>","reasoning":"<brief rationale>"}}
        """
    ).strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def normalize_action_text(text: str) -> str:
    text = strip_code_fences(text).strip().strip("`")
    if text.lower().startswith("json\n"):
        text = text[5:].lstrip()
    text = text.replace("\r", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def extract_json_action(text: str) -> str | None:
    text = strip_code_fences(text).replace("\r", "").strip()
    if text.lower().startswith("json\n"):
        text = text[5:].lstrip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            action = payload.get("action")
            if isinstance(action, str):
                return action.strip()
    return None


def extract_labeled_action(text: str) -> str | None:
    text = strip_code_fences(text).replace("\r", "")
    patterns = [
        r'"action"\s*:\s*"(?P<action>[^"\r\n]+)"',
        r"'action'\s*:\s*'(?P<action>[^'\r\n]+)'",
        r"\baction\s*[:=]\s*(?P<action>[A-Za-z]+\(.*?\)|Wait)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group("action").strip()
    return None


def summarize_model_output(text: str, limit: int = 160) -> str:
    compact = " ".join(strip_code_fences(text).replace("\r", " ").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit - 3]}..."


def match_available_action(text: str, available_actions: Iterable[str]) -> str | None:
    matches: list[str] = []
    for action in available_actions:
        if action in text and action not in matches:
            matches.append(action)
    if len(matches) == 1:
        return matches[0]
    return None


def choose_fallback_action(observation) -> str:
    if observation.recommended_action and observation.recommended_action in observation.available_actions:
        return observation.recommended_action
    return choose_best_available_action(observation)


def parse_action_string(action_text: str) -> AdaptiveLearningSystemAction:
    if action_text == "Wait":
        return AdaptiveLearningSystemAction(action_type="Wait")

    match = re.fullmatch(
        r"(?P<kind>[A-Za-z]+)\((?P<value>[^()]*)\)", action_text)
    if not match:
        raise ValueError(f"Invalid action string: {action_text}")

    kind = match.group("kind")
    value = match.group("value")
    if kind == "AddSubject":
        return AdaptiveLearningSystemAction(action_type="AddSubject", topic_name=value)
    return AdaptiveLearningSystemAction(action_type=kind, concept_id=value)


def _choose_action_blocking(
    client: OpenAI,
    observation,
    step: int,
    action_history: list[str],
) -> tuple[str, str | None]:
    fallback = choose_fallback_action(observation)
    if not observation.available_actions:
        return "Wait", None

    prompt = build_user_prompt(step, observation, action_history[-6:])
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_response = completion.choices[0].message.content or ""
        response_text = (
            extract_json_action(raw_response)
            or extract_labeled_action(raw_response)
            or normalize_action_text(raw_response)
        )
        if response_text in observation.available_actions:
            return response_text, None
        matched_action = match_available_action(
            response_text, observation.available_actions)
        if matched_action:
            return matched_action, None
        return fallback, f"invalid_model_action:{summarize_model_output(raw_response) or 'empty'}"
    except Exception as exc:
        stderr(f"[DEBUG] model_call_failed step={step} error={exc}")
        return fallback, f"model_error:{type(exc).__name__}"


async def choose_action(client: OpenAI, observation, step: int, action_history: list[str]) -> tuple[str, str | None]:
    return await asyncio.to_thread(
        _choose_action_blocking,
        client,
        observation,
        step,
        action_history,
    )


def create_client(api_base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """Build the submission-model client."""

    return OpenAI(
        base_url=api_base_url or API_BASE_URL,
        api_key=api_key or API_KEY,
    )


async def create_env(task_name: str) -> AdaptiveLearningSystemEnv:
    """Connect to a deployed/local server when provided, otherwise use a local Docker image."""

    if ENV_BASE_URL:
        env = AdaptiveLearningSystemEnv(base_url=ENV_BASE_URL)
        await env.connect()
        return env

    return await AdaptiveLearningSystemEnv.from_docker_image(
        LOCAL_IMAGE_NAME,
        provider=_Port7860Provider(),
        env_vars={"ADAPTIVE_LEARNING_TASK": task_name},
    )


async def run_task(client: OpenAI, task_name: str) -> None:
    env = await create_env(task_name)
    rewards: list[float] = []
    action_history: list[str] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        final_score = result.observation.grader_score

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_text, _step_error = await choose_action(
                client, result.observation, step, action_history)
            action = parse_action_string(action_text)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            final_score = result.observation.grader_score

            log_step(
                step=step,
                action=action_text,
                reward=reward,
                done=result.done,
                error=result.observation.last_action_error,
            )

            action_history.append(
                f"step={step} action={action_text} reward={reward:.2f} score={final_score:.3f}"
            )

            if result.done:
                break

        success = final_score >= SUCCESS_SCORE_THRESHOLD
    finally:
        try:
            await env.close()
        except Exception as exc:
            stderr(f"[DEBUG] env_close_failed task={task_name} error={exc}")
        log_end(success=success, steps=steps_taken,
                score=final_score, rewards=rewards)


async def run_all_tasks(client: OpenAI) -> None:
    """Run the configured benchmark task sequence."""

    for task_name in TASKS:
        await run_task(client, task_name)


async def main() -> None:
    client = create_client()
    await run_all_tasks(client)


if __name__ == "__main__":
    asyncio.run(main())
