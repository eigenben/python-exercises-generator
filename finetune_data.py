import pathlib
from typing import Iterable, List, Optional

from exercises import load_all_exercises
from generation import DEFAULT_EXERCISES


def load_finetune_template(prompt_name: str) -> str:
    template_path = pathlib.Path(f"prompts/finetune/{prompt_name}.md")
    return template_path.read_text()


def build_finetune_conversations(
    prompt_name: str,
    exclude_exercise_names: Optional[Iterable[str]] = None,
) -> List[list[dict]]:
    template = load_finetune_template(prompt_name)

    exercises = load_all_exercises()
    if exclude_exercise_names is None:
        exclude_exercise_names = DEFAULT_EXERCISES
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
