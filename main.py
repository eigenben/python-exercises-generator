import argparse
import sys
import re
from pathlib import Path
from exercises import Exercise
from generation import Generator, DEFAULT_EXERCISES
from distillation import StyleDistiller
from helpers import DEFAULT_MODEL
from rich.console import Console
from rich.markdown import Markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python exercises generator CLI"
    )
    subparsers = parser.add_subparsers(
        dest="action", required=True, help="Action to perform"
    )

    # Generate subcommand
    generate_parser = subparsers.add_parser(
        "generate", help="Generate exercise solutions"
    )
    generate_parser.add_argument(
        "--pretty", action="store_true", help="Print result with markdown formatting"
    )
    generate_parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Comma-separated list of example exercise names",
    )
    generate_parser.add_argument(
        "--prompt",
        type=str,
        help="Name of the prompt template to use",
    )
    generate_parser.add_argument(
        "--exercise",
        type=str,
        help="Name of the exercise to use as the problem statement",
    )
    generate_parser.add_argument(
        "--model",
        type=str,
        help="Model to use for generation",
    )
    generate_parser.add_argument(
        "--save",
        type=str,
        nargs='?',
        const='',
        default=None,
        help="Save output to specified file instead of printing to STDOUT (default: output/generations/[prompt]_[exercise]_[model].md)",
    )

    # Distill subcommand
    distill_parser = subparsers.add_parser(
        "distill", help="Distill writing style from example exercises"
    )
    distill_parser.add_argument(
        "--pretty", action="store_true", help="Print result with markdown formatting"
    )
    distill_parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Comma-separated list of example exercise names",
    )
    distill_parser.add_argument(
        "--prompt",
        type=str,
        help="Name of the prompt template to use",
    )
    distill_parser.add_argument(
        "--model",
        type=str,
        help="Model to use for distillation",
    )

    # Finetune subcommand
    finetune_parser = subparsers.add_parser(
        "finetune", help="Finetune a model on exercise solutions"
    )
    finetune_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to finetune (e.g., 'llama-3.3-70b')",
    )
    finetune_parser.add_argument(
        "--prompt",
        type=str,
        default="default",
        help="Name of the prompt template to use",
    )

    args = parser.parse_args()

    if args.action == "generate":
        generator_kwargs = {}
        if args.prompt is not None:
            generator_kwargs['prompt_name'] = args.prompt
        if args.examples is not None:
            example_names = [name.strip() for name in args.examples.split(",")]
            generator_kwargs['example_exercises'] = [Exercise.load(name) for name in example_names]
        if args.model is not None:
            generator_kwargs['model'] = args.model

        generator = Generator(**generator_kwargs)

        if args.exercise:
            problem_statement = Exercise.load(args.exercise).problem_md
        else:
            if sys.stdin.isatty():
                parser.print_help()
                sys.exit(1)
            problem_statement = sys.stdin.read()

        result = generator(problem_statement)

        if args.save is not None:
            if args.save == '':
                # Build default filename
                prompt_name = args.prompt or 'default'
                exercise_name = args.exercise or 'input'
                model_name = args.model or DEFAULT_MODEL
                # Sanitize model name: remove everything before "/", then lowercase and alphanumeric + underscores only
                model_name = model_name.split('/')[-1]
                model_name_sanitized = re.sub(r'[^a-z0-9_]', '_', model_name.lower())
                save_path = Path(f"output/generations/{prompt_name}_{exercise_name}_{model_name_sanitized}.md")
            else:
                save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(result)
        else:
            if args.pretty:
                console = Console()
                console.print(Markdown(result))
            else:
                print(result)

    elif args.action == "distill":
        distiller_kwargs = {}
        if args.prompt is not None:
            distiller_kwargs['prompt_name'] = args.prompt
        if args.examples is not None:
            example_names = [name.strip() for name in args.examples.split(",")]
            distiller_kwargs['example_exercises'] = [Exercise.load(name) for name in example_names]
        if args.model is not None:
            distiller_kwargs['model'] = args.model

        distiller = StyleDistiller(**distiller_kwargs)
        result = distiller()

        if args.pretty:
            console = Console()
            console.print(Markdown(result))
        else:
            print(result)

    elif args.action == "finetune":
        from finetune import Finetuner
        
        finetuner = Finetuner(args.model_name, prompt=args.prompt)
        finetuner.train()
