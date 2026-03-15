# tinyagent

A minimal coding agent that queries a language model, parses bash actions from its output, executes them, and feeds results back — all in a single Python script.

## Installation

```bash
pip install git+ssh://git@github.com/Curt-Park/tinyagent.git
```

### From source

```bash
curl https://mise.run | sh  # Install mise (https://mise.jdx.dev)
mise trust && mise install  # Install Python 3.12 and uv via .mise.toml
uv sync
```

Both methods install `tinyagent` as a CLI command.

### Environment setup

Set your API key before running:

```bash
export TINYAGENT_API_KEY="your-key-here"
```

Or create a `.env` file in your working directory:

```bash
cp .env.example .env
# Add your API key to .env
```

## Usage

```bash
tinyagent "List the files in this directory"
tinyagent "Find all Python files" --max-steps 5
tinyagent "Fix the bug in main.py" --model openrouter/free
tinyagent "Refactor the utils module" --max-context-length 8000
tinyagent "Explain this code" --base-url https://api.openai.com/v1 --model gpt-4o
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
uv sync --extra dev  # Install dev dependencies (ruff, pytest)
ruff check .         # Lint
pytest               # Test
```
