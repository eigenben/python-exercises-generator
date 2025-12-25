# Python Exercises Generator

Tools to generate Python programming exercises based on [TruthfulTechnology/exercises](https://github.com/TruthfulTechnology/exercises).

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
export LLM_BASE_URL="https://your-api-endpoint.com"
export LLM_API_KEY="your-key-here"
```

## Usage

Generate a solution for a problem statement already defined in an exercise:

```bash
uv run main.py --exercise countdown --pretty
```

Generate a solution from a problem statement via stdin:

```bash
echo "Write a function that counts down from n to 0" | uv run main.py
```

### CLI Options

- `--pretty`: Print result with markdown formatting
- `--examples`: Comma-separated list of example exercise names (default: "countdown,count_lines")
- `--prompt`: Name of prompt template to use (default: "two_shot")
- `--exercise`: Name of exercise to use as problem statement
- `--model`: Model to use for generation (default: "gpt-5-mini")

## Exercise Structure

Each exercise is stored in `data/exercises/{exercise_code}/` with the following files:

- `metadata.yml`: Title and description
- `problem.md`: Problem statement
- `solution.md`: Solution
- `test_*.py`: Test code
- `code/*.py`: Optional code samples

## Prompt Templates

Prompt templates are stored in `prompts/*.md` and can include placeholders:

- `{ examples }`: Replaced with formatted example exercises
- `{ problem }`: Replaced with the problem statement

