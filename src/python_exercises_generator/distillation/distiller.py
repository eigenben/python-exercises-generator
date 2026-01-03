from typing import List, Optional

from ..config import get_defaults
from ..exercises import Exercise
from ..integrations.llm import call_llm
from ..prompts import render_prompt

class StyleDistiller:
    def __init__(
        self,
        prompt_name: str = "default",
        example_exercises: Optional[List[Exercise]] = None,
        model: Optional[str] = None,
    ):
        self.prompt_name = prompt_name
        self.model = model
        if example_exercises is None:
            defaults = get_defaults(require=True)
            self.example_exercises = [Exercise.load(name) for name in defaults.distillation_exercises]
        else:
            self.example_exercises = example_exercises

    def prompt(self) -> str:
        examples_text = self._format_examples()
        return render_prompt(
            f"distillation/{self.prompt_name}",
            {"examples": examples_text},
        )

    def __call__(self) -> str:
        prompt_text = self.prompt()
        return call_llm(prompt_text, model=self.model)

    def _format_examples(self) -> str:
        examples = []
        for exercise in self.example_exercises:
            example_text = f"""<example>
<solution>
{exercise.solution_md}
</solution>
</example>"""
            examples.append(example_text)
        return "\n\n".join(examples)
