from __future__ import annotations

from typing import Any, Mapping
import re

from .paths import prompts_dir


def render_prompt(prompt_name: str, vars: Mapping[str, Any]) -> str:
    prompt_path = prompts_dir() / f"{prompt_name}.md"
    template = prompt_path.read_text()

    def replace_var(match: re.Match[str]) -> str:
        var_name = match.group(1).strip()
        return str(vars.get(var_name, match.group(0)))

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace_var, template)
