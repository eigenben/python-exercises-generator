# Python Exercises Generator

Tools to generate Python programming exercise solutions and distill writing styles based on [TruthfulTechnology/exercises](https://github.com/TruthfulTechnology/exercises).

## Installation

```bash
uv sync
```

## Local Setup

After cloning the repository, place any sample exercises you want to use in the `data/exercises` directory. For example:

```bash
git clone git@github.com:TruthfulTechnology/exercises.git
mv exercises/published data/exercises/
rm -rf exercises
```

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

Default model: `mistralai/devstral-2512:free`

## Usage

The CLI has two main subcommands: `generate` and `distill`.

### Generate Subcommand

Generate solutions for programming exercise problems using few-shot learning from example exercises.

Generate a solution from a problem statement defined in an exercise:

```bash
uv run cli.py generate --exercise countdown --pretty
```

Generate a solution from a problem statement via stdin:

```bash
echo "Write a function that counts down from n to 0" | uv run cli.py generate
```

Generate with custom examples and prompt template:

```bash
uv run cli.py generate --exercise flatten --examples "ages,compact,easydict" --prompt with_style_1
```

#### Generate Options

- `--pretty`: Print result with markdown formatting
- `--examples`: Comma-separated list of example exercise names (default: uses Generator defaults: "countdown,count_lines")
- `--prompt`: Name of prompt template to use (default: "default")
- `--exercise`: Name of exercise to use as problem statement (alternative to stdin)
- `--model`: Model to use for generation (default: "mistralai/devstral-2512:free")

### Distill Subcommand

Distill and analyze the writing style from a collection of example exercise solutions.

Distill style from default examples:

```bash
uv run cli.py distill --pretty
```

Distill style from custom examples:

```bash
uv run cli.py distill --examples "ages,compact,flatten,minmax" --pretty
```

#### Distill Options

- `--pretty`: Print result with markdown formatting
- `--examples`: Comma-separated list of example exercise names (default: uses StyleDistiller defaults: "ages,compact,easydict,find_duplicates,flatten,friday,minmax,numeric_range,pluck,reverse_words,transpose,window")
- `--prompt`: Name of prompt template to use (default: "default")
- `--model`: Model to use for distillation (default: "mistralai/devstral-2512:free")

## Architecture

### Core Modules

- **cli.py**: CLI entry point with argument parsing and subcommand routing
- **generation.py**: `Generator` class for creating exercise solutions via LLM
- **distillation.py**: `StyleDistiller` class for analyzing writing style patterns
- **exercises.py**: `Exercise` dataclass for loading and managing exercise data
- **helpers.py**: Utility functions for prompt rendering and LLM API calls

### Exercise Structure

Each exercise is stored in `data/exercises/{exercise_code}/` with the following files:

- `metadata.yml`: Title and description
- `problem.md`: Problem statement
- `solution.md`: Solution
- `test_*.py`: Test code
- `code/*.py`: Optional code samples

### Prompt Templates

Prompt templates are stored in `prompts/{category}/{name}.md` and support variable interpolation using `{{ variable_name }}` syntax.

Available templates:
- `generation/default.md`: Default solution generation prompt
- `generation/with_style_1.md`: Generation with style analysis (variant 1)
- `generation/with_style_2.md`: Generation with style analysis (variant 2)
- `distillation/default.md`: Default style distillation prompt

Common template variables:
- `{{ examples }}`: Replaced with formatted example exercises
- `{{ problem }}`: Replaced with the problem statement (generation only)
