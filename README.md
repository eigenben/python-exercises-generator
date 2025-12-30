# Python Exercises Generator

This is a tool to generate Python programming exercise solutions and distill writing styles of python exercises using large language models (LLMs). It supports few-shot learning from example exercises and can fine-tune models for improved performance. This is primarily an exploratory project to investigate LLM capabilities and limitations of fine tuning.

## Installation

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

Finally, place any sample exercises you want to use in the `data/exercises` directory. Each exercise should be in its own subdirectory two required files: `problem.md` and `solution.md`.

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
- `--save`: Saves the generated solution to `output/generations/<exercise>/<model_name>.md` instead of printing to STDOUT.
- `--model`: Model to use for generation (default: "meta-llama/llama-3.3-70b-instruct:free")
- `--finetuned-model`: Preset name for a fine tuned model to use (overrides `--model` if provided)

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
- `--model`: Model to use for distillation (default: "meta-llama/llama-3.3-70b-instruct:free")
- `--finetuned-model`: Preset name for a fine tuned model to use (overrides `--model` if provided)


### Batch Generate Subcommand

Generate solutions for a batch of exercises in one go (uses threading to parallelize). Saves output to `output/generations/<exercise>/<model_name>.md`:

```bash
uv run python-exercises-generator batch-generate --model meta-llama/llama-3.3-70b-instruct:free
uv run python-exercises-generator batch-generate --model openai/gpt-oss-20b:free
uv run python-exercises-generator batch-generate --model qwen/qwen3-coder-30b-a3b-instruct
```

#### Batch Generate Options

- `--exercises`: Comma-separated list of exercises to generate for (uses `problem.md` in each). Output is saved to `output/generations/<exercise>/<model_name>.md`. Defaults to `DEFAULT_EXERCISES` in `generation.py`
- `--model`: Model to use for generation

### Fine Tune Subcommand

Fine tune a model on a set of example exercises (NVIDIA GPU with CUDA v12.8+ required) based on a preset name (see `FineTuner` class for details).

```bash
uv run python-exercises-generator fine-tune gpt-oss-20b
```

#### Fine Tune Options
- `--prompt`: Name of prompt template to use (default: "default")
- `--save-merged`: Normally we just save a LoRA adapter. With this flag, we also save a merged full model (may be very large).

### Fine Tune Inference Subcommand

Given a model that has already been fine tuned, run inference on a problem statement:

```bash
uv run python-exercises-generator fine-tune-inference gpt-oss-20b --message "Write a python program to play sudoku."
```

To run inference via the `generate` or `batch-generate` subcommands, use the `--finetuned-model` option with the preset name (see above).

### Fine Tuning on Remote GPU Host

Once you have access to a remote GPU host with CUDA v12.8+ and have set up SSH access, rsync or SCP the project directory to the remote host, ensure `uv` is installed, then run the Installation steps above. After that, you can run the fine tuning commands above via SSH. Due to the potentially long runtime of fine tuning jobs, it is recommended to use `tmux` or `screen` to keep the session alive.

### Fine Tuning via Modal.com

Modal.com can be used to run fine tuning jobs on their GPU instances. First, ensure you have a Modal account and have set up the Modal CLI. Then, you can run the fine tuning job using the provided `modal_app.py` script:

```bash
uv run modal run modal_app.py::app.finetune --model gpt-oss-20b
```

You can then run a batch generation job on Modal as well to generate for all default exercises:

```bash
uv run modal run modal_app.py::app.batch_generate --model gpt-oss-20b
```
