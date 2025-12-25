import argparse
import sys
from exercises import Exercise
from generators import Generator
from rich.console import Console
from rich.markdown import Markdown

DEFAULT_EXERCISES = ["countdown", "count_lines"]
DEFAULT_PROMPT = "two_shot"
DEFAULT_MODEL = "gpt-5-mini"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretty", action="store_true", help="Print result with markdown formatting"
    )
    parser.add_argument(
        "--examples",
        type=str,
        default=",".join(DEFAULT_EXERCISES),
        help="Comma-separated list of example exercise names",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Name of the prompt template to use",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default=None,
        help="Name of the exercise to use as the problem statement",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model to use for generation",
    )
    args = parser.parse_args()

    example_names = [name.strip() for name in args.examples.split(",")]
    generator = Generator(
        prompt_name=args.prompt,
        example_exercises=[Exercise.load(name) for name in example_names],
        model=args.model,
    )

    if args.exercise:
        problem_statement = Exercise.load(args.exercise).problem_md
    else:
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        problem_statement = sys.stdin.read()

    result = generator(problem_statement)

    if args.pretty:
        console = Console()
        console.print(Markdown(result))
    else:
        print(result)
