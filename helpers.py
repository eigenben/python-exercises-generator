import pathlib
import re
from typing import Any, Mapping


def render_prompt(prompt_name: str, vars: Mapping[str, Any]) -> str:
    prompt_path = pathlib.Path("prompts") / f"{prompt_name}.md"
    template = prompt_path.read_text()

    # Replace {{ variable_name }} with corresponding values from vars
    def replace_var(match):
        var_name = match.group(1).strip()
        return str(vars.get(var_name, match.group(0)))

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace_var, template)
