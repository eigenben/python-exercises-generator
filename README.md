# Python Exercises Generator

Tools to generate Python programming exercise solutions and distill writing styles based on [TruthfulTechnology/exercises](https://github.com/TruthfulTechnology/exercises).

## Installation

Install the base dependencies:

```bash
uv sync
```

If fine tuning (NVIDIA GPU required with CUDA v12.8+), install the additional finetuning dependencies:

```bash
uv sync --extra finetune
```

## Local Setup

After cloning the repository, place any sample exercises you want to use in the `data/exercises` directory. Each exercise should be in its own subdirectory with the required files (`metadata.yml`, `problem.md`, `solution.md`, etc.).

## Configuration

Set your API keys for LLM integration (defaults to OpenRouter via `OPENROUTER_API_KEY`):

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Alternatively, use a custom LLM endpoint:

```bash
export LLM_BASE_URL="https://your-api-endpoint.com"
export LLM_API_KEY="your-key-here"
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
- `--examples`: Comma-separated list of example exercise names (default: uses Generator defaults: "countdown,count_lines")
- `--prompt`: Name of prompt template to use (default: "default")
- `--exercise`: Name of exercise to use as problem statement (alternative to stdin)
- `--model`: Model to use for generation (default: "meta-llama/llama-3.3-70b-instruct:free")

### Distill Subcommand

Distill and analyze the writing style from a collection of example exercise solutions.

Distill style from default examples (12 canonical exercises):

```bash
uv run python-exercises-generator distill
```


#### Distill Options

- `--pretty`: Print result with markdown formatting
- `--examples`: Comma-separated list of example exercise names (default: uses StyleDistiller defaults: "ages,compact,easydict,find_duplicates,flatten,friday,minmax,numeric_range,pluck,reverse_words,transpose,window")
- `--prompt`: Name of prompt template to use (default: "default")
- `--model`: Model to use for distillation (default: "mistralai/devstral-2512:free")


### Batch Generate Subcommand

Generate solutions for a batch of exercises in one go (uses threading to parallelize):

```bash
uv run python-exercises-generator batch-generate --model meta-llama/llama-3.3-70b-instruct:free
uv run python-exercises-generator batch-generate --model openai/gpt-oss-20b:free
uv run python-exercises-generator batch-generate --model qwen/qwen3-coder-30b-a3b-instruct
```


Generate solutions for finetuned models (see vLLM serving below):

```bash
uv run python-exercises-generator batch-generate --model llama-3.3-70b-instruct-finetuned-python-exercises --base-url "http://155.138.225.139:8000/v1/"
uv run python-exercises-generator batch-generate --model gpt-oss-20b-finetuned-python-exercises --base-url "http://155.138.225.139:8000/v1/"
uv run python-exercises-generator batch-generate --model qwen3-coder-30b-a3b-instruct-finetuned-python-exercises --base-url "http://155.138.225.139:8000/v1/"
```

#### Batch Generate Options

- `--exercises`: Comma-separated list of exercises to generate for (uses `problem.md` in each). Output is saved to `output/generations/<exercise>/<model_name>.md`. Defaults to `DEFAULT_EXERCISES` in `generation.py`
- `--model`: Model to use for generation

#### vLLM Serving

Once you have finetuned models, you can serve them using [vLLM](https://github.com/vllm-project/vllm) to expose an OpenAI-compatible API endpoint that you can subsequently use with `generate` or `batch-generate` by specifying `--base-url`.

First, ensure vllm is installed:

```bash
uv pip install vllm --torch-backend=auto
uv pip install bitsandbytes
```

If `vllm` install fails with lack of `libcudart.so`, install the CUDA toolkit from NVIDIA (`sudo apt install nvidia-cuda-toolkit` on Ubuntu) or set up CUDA via your package manager.

Then, serve a base model (examples below):

```bash
Serving models that have been fine-tuned as LoRA adapters involves serving the original/base model and adding in a LoRA adapter (each of which has a name that can be used as `model_id`). Invoke like this (examples for each fine tune model below):

```bash
uv run vllm serve unsloth/Llama-3.3-70B-Instruct --quantization bitsandbytes --enable-lora --lora-modules llama-3.3-70b-instruct-finetuned-python-exercises=./output/finetuned_models/llama-3.3-70b-instruct-finetuned-python-exercises
uv run vllm serve unsloth/gpt-oss-20b --enable-lora --lora-modules gpt-oss-20b-finetuned-python-exercises=./output/finetuned_models/gpt-oss-20b-finetuned-python-exercises
uv run vllm serve unsloth/Qwen3-Coder-30B-A3B-Instruct --quantization bitsandbytes --enable-lora --lora-modules qwen3-coder-30b-a3b-instruct-finetuned-python-exercises=./output/finetuned_models/qwen3-coder-30b-a3b-instruct-finetuned-python-exercises
```

Note that `vLLM` especially without quantization can be resource intensive; ensure your system has sufficient GPU memory to load the base model along with the LoRA adapter. 80GB of GPU VRAM recommended to be able to serve all models.

## Architecture

### Core Modules

- **cli.py**: CLI entry point with argument parsing and subcommand routing
- **generation.py**: `Generator` class for creating exercise solutions via LLM
- **distillation.py**: `StyleDistiller` class for analyzing writing style patterns
- **exercises.py**: `Exercise` dataclass for loading and managing exercise data
- **helpers.py**: Utility functions for prompt rendering and LLM API calls
