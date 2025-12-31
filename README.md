# Python Exercises Generator

This is a tool to generate Python programming exercise solutions and distill writing styles of Python exercises using large language models (LLMs). It supports few-shot learning from example exercises and can fine-tune models for improved performance. This is primarily an exploratory project to investigate LLM capabilities and limitations of fine-tuning.

Attempts at getting LLMs to generate high-quality programming exercise solutions should start with good prompt engineering and few-shot learning from example exercises. To do that, we can first attempt to have an LLM distill the writing style from a large set of example exercises (using prompts/distillation/default.md by default):

```
uv run python-exercises-generator distill --examples "example1,example2,example3,example4,example5,example6"
```

That distilled style can then be used as part of a prompt such as `prompts/generation/default.md` to generate new solutions for other exercises (as part of generation we also include a few shot examples):

```bash
uv run python-exercises-generator generate --model google/gemma-3-27b-it:free --exercise new_exercise --examples "example1,example2"
```

We can also generate solutions for a batch of exercises in one go using `batch-generate`. Here is an example generating solutions for the default set of exercises and examples defined in `.env` for some common models:

```bash
uv run python-exercises-generator batch-generate --model google/gemma-3-27b-it:free
uv run python-exercises-generator batch-generate --model qwen/qwen3-coder-30b-a3b-instruct
uv run python-exercises-generator batch-generate --model openai/gpt-4o-mini-2024-07-18
uv run python-exercises-generator batch-generate --model google/gemini-3-flash-preview
uv run python-exercises-generator batch-generate --model anthropic/claude-sonnet-4.5
```

We can also attempt to fine-tune a model on a set of example exercises to improve generation quality further (NVIDIA GPU with CUDA v12.8+ required) with the defined presets in `finetuning.py`.

```bash
uv run python-exercises-generator fine-tune qwen3-coder-30b-a3b-instruct
uv run python-exercises-generator fine-tune gemma-3-27b-it
```

These will save LoRA adapters to `output/finetuned_models/<model>-finetuned-python-exercises`. We can then use these fine-tuned models for generation or batch generation via the `--finetuned-model` option by running direct inference with unsloth on the fine-tuned model with LoRA adapter:

```bash
uv run python-exercises-generator batch-generate --finetuned-model qwen3-coder-30b-a3b-instruct
uv run python-exercises-generator batch-generate --finetuned-model gemma-3-27b-it
```

It is also possible to serve the fine tuned models via `vllm` if you `fine-tune` with `--save-merged` to save a merged full model. See `vllm` documentation for details on serving models. Once available at an HTTP endpoint, you can use the `--base-url` and `--api-key` options to run generation/batch generation against the served model.

Finally, we can attempt to fine-tune OpenAI models using their fine-tuning API:

```bash
uv run python-exercises-generator openai-finetune --model gpt-4o-mini-2024-07-18 --wait
```

Once the fine-tuning job is complete, we can use the fine-tuned model ID (e.g. `ft:...`) with the `generate` or `batch-generate` subcommands by passing the model ID via the `--model` option along with the OpenAI base URL and API key:

```bash
uv run python-exercises-generator batch-generate --model ft:your-model-id --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY
```

## Installation

Requires Python 3.12+.

Install the base dependencies:

```bash
uv sync
```

If fine tuning (NVIDIA GPU required with CUDA v12.8+), install the additional finetuning dependencies:

```bash
uv sync --extra finetune
```

Then set up a default set of examples/exercises to use when generating/distilling by editing the `.env` file:

```bash
cp .env.sample .env
```

Finally, place any sample exercises you want to use in the `data/exercises` directory. Each exercise should be in its own subdirectory with:

- `problem.md` (optional): problem statement used for generation.
- `solution.md` (optional): solution used for distillation or finetuning.

Exercise IDs used by CLI flags and `.env` defaults correspond to the subdirectory names under `data/exercises`.

## Configuration

The CLI reads defaults from `.env` and environment variables:

- `DEFAULT_GENERATION_EXAMPLES`: comma-separated example exercise IDs used for few-shot generation.
- `DEFAULT_GENERATION_EXERCISES`: comma-separated exercise IDs used by `batch-generate`.
- `DEFAULT_DISTILLATION_EXERCISES`: comma-separated exercise IDs used for style distillation.
- Prompt templates live in `prompts/generation`, `prompts/distillation`, and `prompts/finetune` (pass the filename stem via `--prompt`).

Set your API keys for LLM integration (defaults to OpenRouter via `OPENROUTER_API_KEY`; `LLM_API_KEY` is also supported):

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Alternatively, use a custom LLM endpoint:

```bash
export LLM_BASE_URL="https://your-api-endpoint.com"
export LLM_API_KEY="your-key-here"
```

If neither `LLM_API_KEY` nor `OPENROUTER_API_KEY` is set, `OPENAI_API_KEY` is used as a fallback for non-`ft:` models.

For OpenAI fine-tuning and fine-tuned inference (also used when you pass an `ft:` model ID):

```bash
export OPENAI_API_KEY="your-openai-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Usage

### Generate Subcommand

Generate solutions for programming exercise problems using few-shot learning from example exercises.

Generate a solution from a problem statement defined in an exercise:

```bash
uv run python-exercises-generator generate --exercise countdown --pretty
```

Generate a solution from a problem statement via stdin:

```bash
echo "Write a function that counts down from n to 0" | uv run python-exercises-generator generate
```

Generate with custom examples:

```bash
uv run python-exercises-generator generate --exercise flatten --examples "ages,compact,easydict"
```

#### Generate Options

- `--pretty`: Print result with markdown formatting
- `--examples`: Comma-separated list of example exercise names (default: uses `DEFAULT_GENERATION_EXAMPLES` from `.env`)
- `--prompt`: Name of prompt template to use (default: "default")
- `--exercise`: Name of exercise to use as problem statement (alternative to stdin)
- `--save`: Saves the generated solution to `output/generations/<exercise>/<model>[_<prompt>].md` (model name is sanitized)
- `--model`: Model to use for generation (default: "meta-llama/llama-3.3-70b-instruct:free")
- `--finetuned-model`: Preset name for a fine tuned model to use (overrides `--model` if provided)
- `--base-url`: Override the LLM base URL (otherwise uses `LLM_BASE_URL` or `OPENROUTER_API_KEY` defaults)
- `--api-key`: Override the LLM API key (otherwise uses `LLM_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY`)

### Distill Subcommand

Distill and analyze the writing style from a collection of example exercise solutions.

Distill style from default examples:

```bash
uv run python-exercises-generator distill
```


#### Distill Options

- `--pretty`: Print result with markdown formatting
- `--examples`: Comma-separated list of example exercise names (default: uses `DEFAULT_DISTILLATION_EXERCISES` from `.env`)
- `--prompt`: Name of prompt template to use (default: "default")
- `--model`: Model to use for distillation (default: "meta-llama/llama-3.3-70b-instruct:free")


### Batch Generate Subcommand

Generate solutions for a batch of exercises in one go (uses threading to parallelize unless using a finetuned model). Saves output to `output/generations/<exercise>/<model>[_<prompt>].md`:

```bash
uv run python-exercises-generator batch-generate --model google/gemma-3-27b-it:free
uv run python-exercises-generator batch-generate --model qwen/qwen3-coder-30b-a3b-instruct
uv run python-exercises-generator batch-generate --model openai/gpt-4o-mini-2024-07-18
```

#### Batch Generate Options

- `--exercises`: Comma-separated list of exercises to generate for (uses `problem.md` in each). Defaults to `DEFAULT_GENERATION_EXERCISES` from `.env`
- `--examples`: Comma-separated list of example exercise names (default: uses `DEFAULT_GENERATION_EXAMPLES` from `.env`)
- `--prompt`: Name of prompt template to use (default: "default")
- `--model`: Model to use for generation (default: "meta-llama/llama-3.3-70b-instruct:free")
- `--finetuned-model`: Preset name for a fine tuned model to use (overrides `--model` if provided)
- `--base-url`: Override the LLM base URL
- `--api-key`: Override the LLM API key

### Fine Tune Subcommand

Fine tune a model on a set of example exercises (NVIDIA GPU with CUDA v12.8+ required) based on a preset name (see `finetune.py` for details). Current presets include:

- `qwen3-coder-30b-a3b-instruct`
- `gemma-3-27b-it`

```bash
uv run python-exercises-generator fine-tune gemma-3-27b-it
```

#### Fine Tune Options
- `--prompt`: Name of prompt template to use (default: "default")
- `--save-merged`: Normally we just save a LoRA adapter. With this flag, we also save a merged full model (may be very large).

### Fine Tune Inference Subcommand

Given a model that has already been fine tuned, run inference on a problem statement:

```bash
uv run python-exercises-generator fine-tune-inference gemma-3-27b-it --message "Write a python program to play sudoku."
```

To run inference via the `generate` or `batch-generate` subcommands, use the `--finetuned-model` option with the preset name (see above).

### Fine Tune Save Merged Subcommand

Merge a LoRA adapter into the base model and save the merged weights:

```bash
uv run python-exercises-generator finetune-save-merged gemma-3-27b-it
```

#### Fine Tune Save Merged Options
- `--output-dir`: Directory to save the merged model (default: `output/finetuned_models/<model>-finetuned-python-exercises-merged`)
- `--save-method`: One of `merged_16bit`, `merged_4bit`, or `lora` (default: `merged_16bit`)
- `--push-to-hub`: Push the merged model to Hugging Face Hub after saving

### Fine Tuning on Remote GPU Host

Once you have access to a remote GPU host with CUDA v12.8+ and have set up SSH access, rsync or SCP the project directory to the remote host, ensure `uv` is installed, then run the Installation steps above. After that, you can run the fine tuning commands above via SSH. Due to the potentially long runtime of fine tuning jobs, it is recommended to use `tmux` or `screen` to keep the session alive.

### Fine Tuning via Modal.com

Modal.com can be used to run fine tuning jobs on their GPU instances. First, ensure you have a Modal account and have set up the Modal CLI. Then, you can run the fine tuning job using the provided `modal_app.py` script:

```bash
uv run modal run modal_app.py::app.finetune --model gemma-3-27b-it
```

You can then run a batch generation job on Modal as well to generate for all default exercises:

```bash
uv run modal run modal_app.py::app.batch_generate --model gemma-3-27b-it
```

### OpenAI Fine Tuning

Prepare JSONL data, upload to OpenAI, and start a fine-tuning job:

```bash
uv run python-exercises-generator openai-finetune --model gpt-4o-mini-2024-07-18 --prompt default --wait
```

Export only (no upload/job):

```bash
uv run python-exercises-generator openai-finetune --model gpt-4o-mini-2024-07-18 --export-only
```

Check job status or wait on a job:

```bash
uv run python-exercises-generator openai-finetune-status --job-id ftjob_... --watch
```

#### OpenAI Fine Tune Options
- `--prompt`: Prompt template name (default: "default")
- `--model`: Base OpenAI model to fine-tune (required)
- `--validation-split`: Fraction of examples for validation (default: 0.1)
- `--seed`: Random seed for the train/validation split (default: 3407)
- `--suffix`: Optional suffix for the fine-tuned model name
- `--output-dir`: Directory for JSONL files (default: `output/openai_finetune/<prompt>`)
- `--export-only`: Only export JSONL files without upload/job creation
- `--wait`: Wait for the fine-tuning job to complete
- `--base-url`: Override OpenAI base URL
- `--api-key`: Override OpenAI API key

#### OpenAI Fine Tune Status Options
- `--job-id`: Fine-tuning job ID (required)
- `--watch`: Poll the job until it completes
- `--interval`: Polling interval in seconds (default: 30)
- `--timeout`: Optional timeout in seconds
- `--base-url`: Override OpenAI base URL
- `--api-key`: Override OpenAI API key

### OpenAI Fine-Tuned Inference

Once you have a fine-tuned model ID (e.g. `ft:...`), run inference using `generate` or `batch-generate` with the `--model` and `--base-url` options:

```bash
uv run python-exercises-generator generate --model ft:your-model-id --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY
uv run python-exercises-generator batch-generate --model ft:your-model-id --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY
```

