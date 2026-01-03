from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List, Optional

from dotenv import load_dotenv

from .paths import project_root


DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


@dataclass(frozen=True)
class Defaults:
    generation_examples: List[str]
    generation_exercises: List[str]
    distillation_exercises: List[str]


def load_env() -> None:
    env_path = project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def get_defaults(require: bool = False) -> Defaults:
    generation_examples = _parse_csv(os.environ.get("DEFAULT_GENERATION_EXAMPLES"))
    generation_exercises = _parse_csv(os.environ.get("DEFAULT_GENERATION_EXERCISES"))
    distillation_exercises = _parse_csv(os.environ.get("DEFAULT_DISTILLATION_EXERCISES"))

    missing = []
    if not generation_examples:
        missing.append("DEFAULT_GENERATION_EXAMPLES")
    if not generation_exercises:
        missing.append("DEFAULT_GENERATION_EXERCISES")
    if not distillation_exercises:
        missing.append("DEFAULT_DISTILLATION_EXERCISES")

    if require and missing:
        raise RuntimeError(
            "Missing required environment variables: "
            f"{', '.join(missing)}. "
            "Please copy .env.sample to .env and configure it."
        )

    return Defaults(
        generation_examples=generation_examples or [],
        generation_exercises=generation_exercises or [],
        distillation_exercises=distillation_exercises or [],
    )
