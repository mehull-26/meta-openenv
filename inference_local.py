"""Local helper for running the benchmark with an OpenAI-compatible local or cloud model."""

from __future__ import annotations

import asyncio
import os

try:
    from . import inference
except ImportError:
    import inference

LOCAL_API_BASE_URL = os.getenv(
    "LOCAL_API_BASE_URL", "http://localhost:11434/v1")
LOCAL_API_KEY = (
    os.getenv("LOCAL_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
    or "ollama"
)
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "gpt-oss-120b-cloud")
LOCAL_ENV_BASE_URL = os.getenv("LOCAL_ENV_BASE_URL")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


async def main() -> None:
    original_model_name = inference.MODEL_NAME
    original_api_base_url = inference.API_BASE_URL
    original_api_key = inference.API_KEY
    original_env_base_url = inference.ENV_BASE_URL
    original_image_name = inference.LOCAL_IMAGE_NAME

    try:
        inference.MODEL_NAME = LOCAL_MODEL_NAME
        inference.API_BASE_URL = LOCAL_API_BASE_URL
        inference.API_KEY = LOCAL_API_KEY

        if LOCAL_ENV_BASE_URL:
            inference.ENV_BASE_URL = LOCAL_ENV_BASE_URL
        if LOCAL_IMAGE_NAME:
            inference.LOCAL_IMAGE_NAME = LOCAL_IMAGE_NAME

        client = inference.create_client(
            api_base_url=LOCAL_API_BASE_URL,
            api_key=LOCAL_API_KEY,
        )
        await inference.run_all_tasks(client)
    finally:
        inference.MODEL_NAME = original_model_name
        inference.API_BASE_URL = original_api_base_url
        inference.API_KEY = original_api_key
        inference.ENV_BASE_URL = original_env_base_url
        inference.LOCAL_IMAGE_NAME = original_image_name


if __name__ == "__main__":
    asyncio.run(main())
