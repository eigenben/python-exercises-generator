from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from ..config import DEFAULT_MODEL


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    if model is None:
        model = DEFAULT_MODEL

    if base_url is None:
        if model and model.startswith("ft:"):
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        else:
            base_url = os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1"

    if api_key is None:
        if model and model.startswith("ft:"):
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = (
                os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

    client = OpenAI(base_url=base_url, api_key=api_key or "")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    message = response.choices[0].message
    return message.content if message.content is not None else message.reasoning
