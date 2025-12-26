import os
from typing import List
from openai import OpenAI
from exercises import Exercise
from helpers import render_prompt

DEFAULT_MODEL = "mistralai/devstral-2512:free"

class Generator:
    def __init__(
        self,
        prompt_name: str,
        example_exercises: List[Exercise],
        model: str = DEFAULT_MODEL,
    ):
        self.prompt_name = prompt_name
        self.example_exercises = example_exercises
        self.model = model

    def prompt(self, problem_statement: str) -> str:
        examples_text = self._format_examples()
        return render_prompt(
            f"generation/{self.prompt_name}",
            {"examples": examples_text, "problem": problem_statement},
        )

    def __call__(self, problem_statement: str) -> str:
        if "LLM_BASE_URL" in os.environ:
            base_url = os.environ["LLM_BASE_URL"]
            api_key = os.environ["LLM_API_KEY"]
        else:
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.environ["OPENROUTER_API_KEY"]

        client = OpenAI(base_url=base_url, api_key=api_key)
        prompt_text = self.prompt(problem_statement)
        response = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt_text}]
        )
        return response.choices[0].message.content

    def _format_examples(self) -> str:
        examples = []
        for exercise in self.example_exercises:
            example_text = f"""<example>
<problem>
{exercise.problem_md}
</problem>
<solution>
{exercise.solution_md}
</solution>
</example>"""
            examples.append(example_text)
        return "\n\n".join(examples)
