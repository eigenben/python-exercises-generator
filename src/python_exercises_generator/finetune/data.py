import pathlib
from typing import Iterable, List, Optional

from ..config import get_defaults
from ..exercises import load_all_exercises
from ..paths import prompts_dir


def load_finetune_template(prompt_name: str) -> str:
    template_path = prompts_dir() / "finetune" / f"{prompt_name}.md"
    return template_path.read_text()


def build_finetune_conversations(
    prompt_name: str,
    exclude_exercise_names: Optional[Iterable[str]] = None,
) -> List[list[dict]]:
    template = load_finetune_template(prompt_name)

    exercises = load_all_exercises()
    if exclude_exercise_names is None:
        defaults = get_defaults(require=True)
        exclude_exercise_names = defaults.generation_exercises
    exclude_set = set(exclude_exercise_names)

    conversations = []
    for exercise in exercises:
        if not exercise.problem_md:
            continue
        if exercise.name in exclude_set:
            continue
        user_message = template.replace("{{ problem }}", exercise.problem_md)
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": exercise.solution_md},
        ]
        conversations.append(conversation)

    return conversations
