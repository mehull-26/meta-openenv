"""Cross-platform pre-submission validator for the adaptive learning project."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_DIR = SCRIPT_DIR
INFERENCE_SCRIPT = SCRIPT_DIR / "inference.py"
DEFAULT_IMAGE = os.getenv("LOCAL_IMAGE_NAME") or "adaptive_learning_system-env:latest"
TASKS = [
    task.strip()
    for task in os.getenv(
        "ADAPTIVE_LEARNING_TASKS",
        "task1_review,task2_add_subject,task3_triage",
    ).split(",")
    if task.strip()
]

STEP_PATTERN = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>.+) reward=(?P<reward>-?\d+\.\d{2}) "
    r"done=(?P<done>true|false) error=(?P<error>.+)$"
)
START_PATTERN = re.compile(r"^\[START\] task=(?P<task>[^\s]+) env=(?P<env>[^\s]+) model=(?P<model>.+)$")
END_PATTERN = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) score=(?P<score>\d+\.\d{3}) rewards=(?P<rewards>.*)$"
)


def log(message: str) -> None:
    print(message, flush=True)


def find_command(local_paths: list[Path], fallback: str) -> list[str]:
    for local_path in local_paths:
        if local_path.exists():
            return [str(local_path)]
    return [fallback]


def find_venv_command(executable_name: str, fallback: str) -> list[str]:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    candidates = [
        SCRIPT_DIR / ".venv" / scripts_dir / executable_name,
        REPO_ROOT / ".venv" / scripts_dir / executable_name,
    ]
    return find_command(candidates, fallback)


def _decode_output(raw: bytes | None) -> str:
    """Decode subprocess output without crashing on Windows codepage mismatches."""

    if not raw:
        return ""
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
    )
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=_decode_output(result.stdout),
        stderr=_decode_output(result.stderr),
    )


def ensure_docker_available() -> None:
    """Fail fast with a clearer message if Docker CLI or daemon is unavailable."""

    version_result = run_command(["docker", "version"])
    combined_output = (version_result.stdout or "") + (version_result.stderr or "")

    if version_result.returncode == 0:
        return

    lowered = combined_output.lower()
    if "docker api" in lowered or "dockerdesktoplinuxengine" in lowered or "daemon" in lowered:
        raise RuntimeError(
            "Docker is installed, but the Docker daemon is not running.\n"
            "Start Docker Desktop (or the Docker service) and wait until it shows as running,\n"
            "then rerun validate_submission.py.\n\n"
            f"Original docker error:\n{combined_output.strip()}"
        )

    raise RuntimeError(
        "Docker CLI is not available. Install Docker Desktop / Docker Engine and ensure `docker` is on PATH.\n\n"
        f"Original docker error:\n{combined_output.strip()}"
    )


def resolve_dockerfile() -> tuple[Path, Path]:
    """Find the Dockerfile path and matching build context for this environment."""

    root_dockerfile = ENV_DIR / "Dockerfile"
    server_dockerfile = ENV_DIR / "server" / "Dockerfile"

    if root_dockerfile.exists():
        return root_dockerfile, ENV_DIR
    if server_dockerfile.exists():
        return server_dockerfile, ENV_DIR

    raise RuntimeError(f"No Dockerfile found in {ENV_DIR} or {server_dockerfile.parent}")


def ping_space(space_url: str) -> None:
    reset_url = f"{space_url.rstrip('/')}/reset"
    request = urllib.request.Request(
        reset_url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"/reset returned HTTP {response.status}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {reset_url}: {exc}") from exc


def validate_inference_output(output: str) -> None:
    starts = 0
    ends = 0
    task_names: list[str] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if START_PATTERN.match(line):
            starts += 1
            task_names.append(START_PATTERN.match(line).group("task"))  # type: ignore[union-attr]
            continue
        step_match = STEP_PATTERN.match(line)
        if step_match:
            reward = float(step_match.group("reward"))
            if not 0.0 <= reward <= 1.0:
                raise RuntimeError(f"Step reward out of range [0,1]: {line}")
            continue
        end_match = END_PATTERN.match(line)
        if end_match:
            ends += 1
            score = float(end_match.group("score"))
            if not 0.0 <= score <= 1.0:
                raise RuntimeError(f"Task score out of range [0,1]: {line}")
            continue
        raise RuntimeError(f"Unexpected stdout line from inference.py: {line}")

    if starts != len(TASKS):
        raise RuntimeError(f"Expected {len(TASKS)} [START] lines, found {starts}")
    if ends != len(TASKS):
        raise RuntimeError(f"Expected {len(TASKS)} [END] lines, found {ends}")
    if task_names != TASKS:
        raise RuntimeError(f"Task order mismatch. Expected {TASKS}, got {task_names}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-url", help="Optional deployed HF Space URL to ping before local checks.")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE, help="Docker image tag to build and run.")
    args = parser.parse_args()

    python_name = "python.exe" if os.name == "nt" else "python"
    python_cmd = find_venv_command(python_name, sys.executable)
    openenv_cmd = [*python_cmd, "-m", "openenv.cli"]

    if args.space_url:
        log("Step 1/4: Pinging deployed Space")
        ping_space(args.space_url)
        log("PASS: Space responds to /reset")

    log("Step 2/4: Running openenv validate")
    validate_result = run_command(openenv_cmd + ["validate", str(ENV_DIR), "-v"], cwd=ENV_DIR)
    if validate_result.returncode != 0:
        raise RuntimeError(validate_result.stdout + validate_result.stderr)
    log(validate_result.stdout.strip() or "PASS: openenv validate")

    log("Step 3/4: Building Docker image")
    ensure_docker_available()
    dockerfile_path, build_context = resolve_dockerfile()
    docker_result = run_command(
        ["docker", "build", "-t", args.image_name, "-f", str(dockerfile_path), str(build_context)],
        cwd=ENV_DIR,
    )
    if docker_result.returncode != 0:
        raise RuntimeError(docker_result.stdout + docker_result.stderr)
    log(f"PASS: docker build succeeded using {dockerfile_path}")

    log("Step 4/4: Running inference.py")
    inference_env = os.environ.copy()
    inference_env.setdefault("LOCAL_IMAGE_NAME", args.image_name)
    inference_env.setdefault("ADAPTIVE_LEARNING_TASKS", ",".join(TASKS))
    inference_env.setdefault("ENV_BASE_URL", "")
    inference_result = run_command(python_cmd + [str(INFERENCE_SCRIPT)], cwd=ENV_DIR, env=inference_env)
    if inference_result.returncode != 0:
        raise RuntimeError(inference_result.stdout + inference_result.stderr)
    validate_inference_output(inference_result.stdout)
    log("PASS: inference.py emitted valid Scaler/OpenEnv logs")
    log("Validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
