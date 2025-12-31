import os
import pathlib
import re
from typing import Any, Mapping, Optional
from openai import OpenAI

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def render_prompt(prompt_name: str, vars: Mapping[str, Any]) -> str:
    prompt_path = pathlib.Path("prompts") / f"{prompt_name}.md"
    template = prompt_path.read_text()

    # Replace {{ variable_name }} with corresponding values from vars
    def replace_var(match):
        var_name = match.group(1).strip()
        return str(vars.get(var_name, match.group(0)))

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace_var, template)


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Call LLM with the given prompt and return the response content."""
    if model is None:
        model = DEFAULT_MODEL

    # Use provided base_url/api_key if given, otherwise fall back to environment variables
    if base_url is None:
        if model and model.startswith("ft:"):
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        else:
            base_url = os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1"

    if api_key is None:
        if model and model.startswith("ft:"):
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    client = OpenAI(base_url=base_url, api_key=api_key or "")
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    message = response.choices[0].message
    return message.content if message.content is not None else message.reasoning
