import os
import pathlib
import re
from typing import Any, Mapping, Optional
from openai import OpenAI

DEFAULT_MODEL = "mistralai/devstral-2512:free"


def render_prompt(prompt_name: str, vars: Mapping[str, Any]) -> str:
    prompt_path = pathlib.Path("prompts") / f"{prompt_name}.md"
    template = prompt_path.read_text()

    # Replace {{ variable_name }} with corresponding values from vars
    def replace_var(match):
        var_name = match.group(1).strip()
        return str(vars.get(var_name, match.group(0)))

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace_var, template)


def call_llm(prompt: str, model: Optional[str] = None) -> str:
    """Call LLM with the given prompt and return the response content."""
    if model is None:
        model = DEFAULT_MODEL

    if "LLM_BASE_URL" in os.environ:
        base_url = os.environ["LLM_BASE_URL"]
        api_key = os.environ["LLM_API_KEY"]
    else:
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ["OPENROUTER_API_KEY"]

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
