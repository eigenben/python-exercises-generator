from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

from ..config import get_defaults
from ..exercises import Exercise
from ..integrations.llm import call_llm
from ..paths import output_dir
from ..prompts import render_prompt


def save_generation(
    content: str,
    prompt_name: str,
    exercise_name: str,
    model: Optional[str] = None,
) -> Path:
    """Save generated content to a file.

    Args:
        content: The generated content to save
        prompt_name: Name of the prompt template used
        exercise_name: Name of the exercise
        model: Model name (optional, uses DEFAULT_MODEL if not provided)

    Returns:
        Path to the saved file
    """
    from ..config import DEFAULT_MODEL
    model_name = model or DEFAULT_MODEL
    # Sanitize model name: remove everything before "/", then lowercase and alphanumeric + underscores only
    model_name = model_name.split('/')[-1]
    model_name_sanitized = re.sub(r'[^a-z0-9_]', '_', model_name.lower())

    # Build filename: if prompt is "default", don't include it
    if prompt_name == "default":
        filename = f"{model_name_sanitized}.md"
    else:
        filename = f"{model_name_sanitized}_{prompt_name}.md"

    save_path = output_dir() / "generations" / exercise_name / filename

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(content)
    return save_path

class Generator:
    def __init__(
        self,
        prompt_name: str = "default",
        example_exercises: Optional[List[Exercise]] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        finetuned_model: Optional[str] = None,
    ):
        self.prompt_name = prompt_name
        self.finetuned_model = finetuned_model
        # If finetuned_model is specified but model is not, default model name
        if finetuned_model is not None and model is None:
            self.model = f"{finetuned_model}-finetuned-python-exercises"
        else:
            self.model = model
        self.base_url = base_url
        self.api_key = api_key
        if example_exercises is None:
            defaults = get_defaults(require=True)
            self.example_exercises = [Exercise.load(name) for name in defaults.generation_examples]
        else:
            self.example_exercises = example_exercises

    def prompt(self, problem_statement: str) -> str:
        examples_text = self._format_examples()
        return render_prompt(
            f"generation/{self.prompt_name}",
            {"examples": examples_text, "problem": problem_statement},
        )

    def __call__(
        self,
        problem_statement: str,
        save: bool = False,
        exercise_name: Optional[str] = None,
    ) -> str:
        prompt_text = self.prompt(problem_statement)

        if self.finetuned_model is not None:
            from ..finetune import Finetuner
            finetuner = Finetuner(self.finetuned_model)
            result = finetuner.inference(prompt_text)
        else:
            result = call_llm(
                prompt_text,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
            )

        if save:
            if exercise_name is None:
                exercise_name = "input"
            save_generation(
                content=result,
                prompt_name=self.prompt_name,
                exercise_name=exercise_name,
                model=self.model,
            )

        return result

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


class BatchGenerator:
    def __init__(
        self,
        prompt_name: str = "default",
        example_exercises: Optional[List[Exercise]] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        finetuned_model: Optional[str] = None,
    ):
        self.prompt_name = prompt_name
        self.example_exercises = example_exercises
        self.finetuned_model = finetuned_model
        # If finetuned_model is specified but model is not, default model name
        if finetuned_model is not None and model is None:
            self.model = f"{finetuned_model}-finetuned-python-exercises"
        else:
            self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def __call__(self, exercise_names: List[str]) -> List[dict]:
        """Generate solutions for multiple exercises concurrently.

        Args:
            exercise_names: List of exercise names to generate solutions for

        Returns:
            List of dicts containing exercise name, output path, and success status
        """
        results = []

        # If using a finetuned model, create and load a single instance
        finetuner = None
        if self.finetuned_model is not None:
            from ..finetune import Finetuner
            finetuner = Finetuner(self.finetuned_model)
            finetuner.load_finetuned_model_for_inference()

        def generate_and_save(exercise_name: str) -> dict:
            """Generate solution for a single exercise and save it."""
            try:
                # Load the exercise
                exercise = Exercise.load(exercise_name)

                # Create a generator instance
                generator = Generator(
                    prompt_name=self.prompt_name,
                    example_exercises=self.example_exercises,
                    model=self.model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                )

                # Generate the prompt
                prompt_text = generator.prompt(exercise.problem_md)

                # Generate the solution using finetuner or regular LLM
                if finetuner is not None:
                    result = finetuner.inference(prompt_text)
                else:
                    result = call_llm(
                        prompt_text,
                        model=self.model,
                        base_url=self.base_url,
                        api_key=self.api_key,
                    )

                # Save the result
                save_path = save_generation(
                    content=result,
                    prompt_name=self.prompt_name,
                    exercise_name=exercise_name,
                    model=self.finetuned_model or self.model,
                )

                return {
                    "exercise": exercise_name,
                    "path": str(save_path),
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "exercise": exercise_name,
                    "path": None,
                    "success": False,
                    "error": str(e),
                }

        # Run sequentially if using finetuned model, in parallel otherwise
        if self.finetuned_model is not None:
            # Sequential execution for finetuned models
            for exercise_name in exercise_names:
                result = generate_and_save(exercise_name)
                results.append(result)
        else:
            # Use ThreadPoolExecutor to run generations concurrently
            with ThreadPoolExecutor() as executor:
                # Submit all tasks
                future_to_exercise = {
                    executor.submit(generate_and_save, name): name
                    for name in exercise_names
                }

                # Collect results as they complete
                for future in as_completed(future_to_exercise):
                    results.append(future.result())

        return results
