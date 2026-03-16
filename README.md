# tinyagent

A coding agent in a single Python script under 150 lines for educational purpose.
It calls an LLM, executes bash commands via function-calling, and loops until the task is done.
Works with any OpenAI-compatible API (OpenRouter, OpenAI, vLLM, Ollama, etc.).

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
export TINYAGENT_API_KEY="your-key-here"  # Default: OpenRouter API key is required
```

Or create a `.env` file in your working directory:

```bash
cp .env.example .env
# Add your API key to .env
```

## Usage

```bash
tinyagent "List the files in this directory"
tinyagent "Fix the bug in main.py" --config path/to/custom.yaml
tinyagent "Explain this repo" --model google/gemini-2.0-flash-001  # override model
tinyagent "Refactor utils.py" --max-steps 10 --command-timeout 60  # override limits
```

By default, tinyagent uses the `openrouter/free` model, which requires no payment but may produce poor results (e.g., garbled tool calls or incomplete answers).
For better performance, use `--model` to select a higher-quality model.

The default config is bundled at `config/default.yaml`.
Use `--config` to override with a custom YAML file, or use CLI flags to override individual settings.

Run `tinyagent --help` for all available options.

## Configuration

All settings are defined in a YAML config file (`config/default.yaml`):

| Key                  | Description                                      |
|----------------------|--------------------------------------------------|
| `system_prompt`      | System message for the LLM                       |
| `instance_template`  | User message template (`{{task}}` is replaced)   |
| `compact_prompt`     | Prompt used to summarize older conversation turns |
| `model`              | Model name                                       |
| `base_url`           | LLM API base URL                                 |
| `max_steps`          | Maximum agent loop iterations                    |
| `max_context_length` | Estimated token budget for context compaction     |
| `command_timeout`    | Seconds before a command is killed                |

## Features

- OpenAI function-calling (`tools` param) for structured bash execution
- YAML-based config — prompts, model, and defaults are not hardcoded
- Instance template with few-shot examples for the task prompt
- Context compaction — summarizes older turns when approaching the token budget
- Trajectory logging to JSON after each run

## Development

```bash
uv sync --extra dev  # Install dev dependencies (ruff, pytest)
ruff check .         # Lint
pytest               # Test
```

## References

- [SWE-agent](https://github.com/SWE-agent/SWE-agent)
- [live-swe-agent](https://github.com/OpenAutoCoder/live-swe-agent)
