import argparse
import sys
from exercises import Exercise
from generation import Generator, BatchGenerator, DEFAULT_EXERCISES
from distillation import StyleDistiller
from helpers import DEFAULT_MODEL
from rich.console import Console
from rich.markdown import Markdown


def main() -> None:
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
        "--finetuned-model",
        type=str,
        help="Finetuned model to use for generation (overrides --model)",
    )
    generate_parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to output/generations/[exercise]/[model].md instead of printing to STDOUT",
    )
    generate_parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for the LLM API",
    )
    generate_parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the LLM API",
    )

    # Batch generate subcommand
    batch_generate_parser = subparsers.add_parser(
        "batch-generate", help="Generate solutions for multiple exercises concurrently"
    )
    batch_generate_parser.add_argument(
        "--exercises",
        type=str,
        default=None,
        help="Comma-separated list of exercise names to generate solutions for (defaults to standard exercise set)",
    )
    batch_generate_parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Comma-separated list of example exercise names",
    )
    batch_generate_parser.add_argument(
        "--prompt",
        type=str,
        help="Name of the prompt template to use",
    )
    batch_generate_parser.add_argument(
        "--model",
        type=str,
        help="Model to use for generation",
    )
    batch_generate_parser.add_argument(
        "--finetuned-model",
        type=str,
        help="Finetuned model to use for generation (overrides --model)",
    )
    batch_generate_parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for the LLM API",
    )
    batch_generate_parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the LLM API",
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
    finetune_parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save merged model (LoRA + base) after training in addition to LoRA adapter",
    )

    # Finetune inference subcommand
    finetune_inference_parser = subparsers.add_parser(
        "finetune-inference", help="Run inference with a finetuned model"
    )
    finetune_inference_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the finetuned model to use for inference",
    )
    finetune_inference_parser.add_argument(
        "--message",
        type=str,
        required=True,
        help="The message/prompt to send to the model",
    )

    # Finetune save merged subcommand
    finetune_save_merged_parser = subparsers.add_parser(
        "finetune-save-merged", help="Save a finetuned model with LoRA adapter merged into base model"
    )
    finetune_save_merged_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the finetuned model to save merged (e.g., 'llama-3.3-70b-instruct')",
    )
    finetune_save_merged_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the merged model (defaults to output/finetuned_models/{model_name}-finetuned-python-exercises-merged)",
    )
    finetune_save_merged_parser.add_argument(
        "--save-method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit", "lora"],
        help="Method to use for saving (default: merged_16bit, recommended for vLLM)",
    )
    finetune_save_merged_parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the merged model to Hugging Face Hub after saving",
    )

    # OpenAI finetune subcommand
    openai_finetune_parser = subparsers.add_parser(
        "openai-finetune", help="Fine-tune an OpenAI model on exercise solutions"
    )
    openai_finetune_parser.add_argument(
        "--prompt",
        type=str,
        default="default",
        help="Name of the finetuning prompt template to use",
    )
    openai_finetune_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base OpenAI model to fine-tune (e.g. gpt-4o-mini-2024-07-18)",
    )
    openai_finetune_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of examples to use for validation (default: 0.1)",
    )
    openai_finetune_parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for train/validation split",
    )
    openai_finetune_parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix for the fine-tuned model name",
    )
    openai_finetune_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the JSONL dataset files",
    )
    openai_finetune_parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export JSONL files without uploading or starting a fine-tune job",
    )
    openai_finetune_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the fine-tuning job to complete",
    )
    openai_finetune_parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for the OpenAI API",
    )
    openai_finetune_parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the OpenAI API",
    )

    # OpenAI finetune status subcommand
    openai_finetune_status_parser = subparsers.add_parser(
        "openai-finetune-status", help="Check or wait on an OpenAI fine-tuning job"
    )
    openai_finetune_status_parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="Fine-tuning job ID",
    )
    openai_finetune_status_parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll the job until it completes",
    )
    openai_finetune_status_parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30)",
    )
    openai_finetune_status_parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds",
    )
    openai_finetune_status_parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for the OpenAI API",
    )
    openai_finetune_status_parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the OpenAI API",
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
        if args.finetuned_model is not None:
            generator_kwargs['finetuned_model'] = args.finetuned_model
        if args.base_url is not None:
            generator_kwargs['base_url'] = args.base_url
        if args.api_key is not None:
            generator_kwargs['api_key'] = args.api_key

        generator = Generator(**generator_kwargs)

        if args.exercise:
            problem_statement = Exercise.load(args.exercise).problem_md
            exercise_name = args.exercise
        else:
            if sys.stdin.isatty():
                parser.print_help()
                sys.exit(1)
            problem_statement = sys.stdin.read()
            exercise_name = None

        result = generator(
            problem_statement=problem_statement,
            save=args.save,
            exercise_name=exercise_name,
        )

        if not args.save:
            if args.pretty:
                console = Console()
                console.print(Markdown(result))
            else:
                print(result)

    elif args.action == "batch-generate":
        batch_generator_kwargs = {}
        if args.prompt is not None:
            batch_generator_kwargs['prompt_name'] = args.prompt
        else:
            batch_generator_kwargs['prompt_name'] = 'default'
        if args.examples is not None:
            example_names = [name.strip() for name in args.examples.split(",")]
            batch_generator_kwargs['example_exercises'] = [Exercise.load(name) for name in example_names]
        if args.model is not None:
            batch_generator_kwargs['model'] = args.model
        if args.finetuned_model is not None:
            batch_generator_kwargs['finetuned_model'] = args.finetuned_model
        if args.base_url is not None:
            batch_generator_kwargs['base_url'] = args.base_url
        if args.api_key is not None:
            batch_generator_kwargs['api_key'] = args.api_key

        batch_generator = BatchGenerator(**batch_generator_kwargs)

        if args.exercises is not None:
            exercise_names = [name.strip() for name in args.exercises.split(",")]
        else:
            exercise_names = DEFAULT_EXERCISES
        results = batch_generator(exercise_names)

        # Print summary of results
        console = Console()
        console.print("\n[bold]Batch Generation Results:[/bold]\n")
        for result in results:
            if result["success"]:
                console.print(f"[green]✓[/green] {result['exercise']}: {result['path']}")
            else:
                console.print(f"[red]✗[/red] {result['exercise']}: {result['error']}")

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

        if args.save_merged:
            console = Console()
            console.print("\n[bold blue]Saving merged model...[/bold blue]")
            output_dir = finetuner.save_merged_model()
            console.print(f"[bold green]✓ Merged model saved successfully![/bold green]")
            console.print(f"[dim]  • Location: {output_dir}[/dim]")
            console.print(f"\n[bold blue]Deploy with vLLM:[/bold blue]")
            console.print(f"[dim]  vllm serve {output_dir}[/dim]\n")

    elif args.action == "finetune-inference":
        from finetune import Finetuner

        finetuner = Finetuner(args.model_name)
        print(finetuner.inference(args.message))

    elif args.action == "finetune-save-merged":
        from finetune import Finetuner

        console = Console()
        console.print(f"\n[bold blue]Loading finetuned model:[/bold blue] [yellow]{args.model_name}[/yellow]")

        finetuner = Finetuner(args.model_name)
        finetuner.load_finetuned_model()

        console.print("[green]✓ Finetuned model loaded[/green]")

        output_dir = finetuner.save_merged_model(
            output_dir=args.output_dir,
            save_method=args.save_method,
            push_to_hub=args.push_to_hub
        )

        console.print(f"\n[bold green]✓ Merged model saved successfully![/bold green]")
        console.print(f"[dim]  • Location: {output_dir}[/dim]")
        console.print(f"\n[bold blue]Deploy with vLLM:[/bold blue]")
        console.print(f"[dim]  vllm serve {output_dir}[/dim]\n")

    elif args.action == "openai-finetune":
        from openai_finetune import OpenAIFinetuner

        console = Console()
        console.print("\n[bold blue]Preparing OpenAI fine-tuning dataset...[/bold blue]")
        finetuner = OpenAIFinetuner(
            prompt_name=args.prompt,
            model=args.model,
            output_dir=args.output_dir,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        paths, stats = finetuner.prepare_dataset(
            validation_split=args.validation_split,
            seed=args.seed,
        )
        console.print(f"[green]✓[/green] Total examples: {stats.total_examples}")
        console.print(f"[dim]  • Training: {stats.training_examples}[/dim]")
        console.print(f"[dim]  • Validation: {stats.validation_examples}[/dim]")
        console.print(f"[dim]  • Output dir: {paths.output_dir}[/dim]")

        if args.export_only:
            console.print("[yellow]Dataset export only; skipping upload and job creation.[/yellow]")
            return

        console.print("\n[bold blue]Uploading training file...[/bold blue]")
        training_file_id, validation_file_id = finetuner.upload_dataset(paths)
        console.print(f"[green]✓[/green] Training file ID: {training_file_id}")

        if paths.validation_file is not None:
            console.print(f"[green]✓[/green] Validation file ID: {validation_file_id}")

        console.print("\n[bold blue]Creating fine-tuning job...[/bold blue]")
        job = finetuner.create_job(
            training_file_id=training_file_id,
            validation_file_id=validation_file_id,
            suffix=args.suffix,
        )
        console.print(f"[green]✓[/green] Job ID: {job.id}")
        console.print(f"[dim]  • Status: {job.status}[/dim]")

        metadata_path = finetuner.save_job_metadata(
            job=job,
            training_file_id=training_file_id,
            validation_file_id=validation_file_id,
            paths=paths,
        )
        console.print(f"[dim]  • Metadata: {metadata_path}[/dim]")

        if args.wait:
            console.print("\n[bold blue]Waiting for job to complete...[/bold blue]")
            job = finetuner.wait_for_job(job.id)
            console.print(f"[bold]Final status:[/bold] {job.status}")
            if getattr(job, "fine_tuned_model", None):
                console.print(f"[green]✓[/green] Fine-tuned model: {job.fine_tuned_model}")
            metadata_path = finetuner.save_job_metadata(
                job=job,
                training_file_id=training_file_id,
                validation_file_id=validation_file_id,
                paths=paths,
            )
            console.print(f"[dim]  • Metadata updated: {metadata_path}[/dim]")

    elif args.action == "openai-finetune-status":
        from openai_finetune import OpenAIFinetuner

        finetuner = OpenAIFinetuner(
            api_key=args.api_key,
            base_url=args.base_url,
        )
        if args.watch:
            job = finetuner.wait_for_job(
                args.job_id,
                poll_interval=args.interval,
                timeout_seconds=args.timeout,
            )
        else:
            job = finetuner.get_job(args.job_id)

        console = Console()
        console.print(f"[bold]Job ID:[/bold] {job.id}")
        console.print(f"[bold]Status:[/bold] {job.status}")
        if getattr(job, "fine_tuned_model", None):
            console.print(f"[green]✓[/green] Fine-tuned model: {job.fine_tuned_model}")

if __name__ == "__main__":
    main()
