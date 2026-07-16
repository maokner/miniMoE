"""Validation and rendering for the committed deterministic prompt suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chat_template import append_reply, build_inference_prompt
from model import EOS_TOKEN_ID


REQUIRED_CATEGORIES = {
    "scientific_completion",
    "narrative_completion",
    "factual_recall",
    "concise_explanation",
    "summarization",
    "practical_instruction",
    "probability",
    "arithmetic",
    "elementary_python",
    "impossible_factual_premise",
    "ambiguity",
    "two_turn_memory",
}


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "prompts.json"


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest = json.loads(Path(path or default_manifest_path()).read_text())
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    prompts = manifest.get("prompts", [])
    if len(prompts) != 12:
        raise ValueError("Prompt manifest must contain exactly 12 cases")
    ids = [prompt.get("id") for prompt in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("Prompt IDs must be unique")
    categories = {prompt.get("category") for prompt in prompts}
    if categories != REQUIRED_CATEGORIES:
        raise ValueError(
            f"Prompt categories differ from required set: {categories ^ REQUIRED_CATEGORIES}"
        )
    raw = [prompt for prompt in prompts if prompt.get("mode") == "raw_base_only"]
    if len(raw) != 2 or any("prompt" not in prompt for prompt in raw):
        raise ValueError("Exactly two raw base-only prompts are required")
    instruction = [prompt for prompt in prompts if prompt.get("mode") == "instruction"]
    if len(instruction) != 10 or any(
        not prompt.get("messages") for prompt in instruction
    ):
        raise ValueError("The remaining ten prompts must use instruction messages")
    generation = manifest.get("generation", {})
    if generation.get("temperature") != 0 or generation.get("max_new_tokens") != 96:
        raise ValueError(
            "Published prompts must use greedy decoding with a 96-token limit"
        )


def render_prompt(prompt: dict[str, Any], tokenizer) -> list[int]:
    if prompt["mode"] == "raw_base_only":
        return [EOS_TOKEN_ID] + tokenizer.encode(prompt["prompt"])

    history: list[int] = []
    messages = prompt["messages"]
    for index in range(0, len(messages) - 1, 2):
        user = messages[index]
        assistant = messages[index + 1]
        if user["role"] != "user" or assistant["role"] != "assistant":
            raise ValueError(f"Invalid role order in prompt {prompt['id']}")
        history = append_reply(
            history, user["content"], assistant["content"], tokenizer
        )
    final = messages[-1]
    if final["role"] != "user":
        raise ValueError(f"Prompt {prompt['id']} must end with a user message")
    return build_inference_prompt(history, final["content"], tokenizer)
