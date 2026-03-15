# tinyagent

A minimal coding agent that queries a language model, parses bash actions from its output, executes them, and feeds results back — all in a single Python script.

## Setup

```bash
curl https://mise.run | sh  # Install mise (https://mise.jdx.dev)
mise trust && mise install  # Install Python 3.12 and uv via .mise.toml
uv sync --extra dev
cp .env.example .env
# Add your OpenRouter API key to .env
```

## Usage

```bash
python run.py "List the files in this directory"
python run.py "Find all Python files" --max-steps 5
python run.py "Fix the bug in main.py" --model openrouter/free
python run.py "Refactor the utils module" --max-context-length 8000
```

## Features

- LLM-driven bash command execution in an agent loop
- Automatic output truncation for long command results
- Context compaction — summarizes older turns when approaching the token budget
- Configurable model, step limit, and context length
- Trajectory logging to JSON after each run

A `trajectory_<timestamp>.json` file is saved after each run.

## Development

```bash
ruff check .         # Lint
pytest               # Test
```
