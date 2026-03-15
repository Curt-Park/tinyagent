"""tinyagent: A minimal coding agent.

Queries a language model, parses bash actions from its output,
executes them, and feeds results back to the model.
"""

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """\
You are a coding agent. You solve tasks by running bash commands.

On each turn reply with EXACTLY:
1. THOUGHT: one sentence describing your plan.
2. A single bash command inside a ```bash fenced block.

When the task is complete, you MUST immediately run:
```bash
exit
```

Rules:
- ONE command per turn. No extra commentary after the code block.
- Never use interactive commands (vim, less, python REPL, etc.).
- Output may be truncated — work with what you see.
- As soon as you have the answer or have completed the task, exit. Do NOT keep exploring.
"""


class TerminatingError(Exception):
    """Raised when the agent should gracefully stop."""


def query_lm(messages: list[dict[str, str]], model: str) -> str:
    """Send messages to the language model and return the response text."""
    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def parse_action(lm_output: str) -> str:
    """Extract a bash command from a ```bash-action``` fenced code block."""
    matches: list[str] = re.findall(
        r"```(?:bash-action|bash)\s*\n?(.*?)\n?```",
        lm_output,
        re.DOTALL,
    )
    if not matches:
        raise TerminatingError("No action block found — task complete.")
    command: str = matches[0].strip()
    if command == "exit":
        raise TerminatingError("exit")
    return command


def execute_action(command: str) -> str:
    """Run a shell command and return its combined stdout/stderr."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        command,
        shell=True,
        text=True,
        env=os.environ,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return result.stdout


def truncate_output(text: str, max_chars: int = 10_000) -> str:
    """Truncate long output, keeping head and tail with a marker."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    removed = len(text) - max_chars
    return f"{text[:half]}\n[...truncated {removed} chars...]\n{text[-half:]}"


def save_trajectory(messages: list[dict[str, str]], path: str) -> None:
    """Write the conversation messages to a JSON file."""
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)


def main() -> None:
    """Run the agent loop: query → parse → execute → feed back."""
    parser = argparse.ArgumentParser(description="tinyagent — a minimal coding agent")
    parser.add_argument("task", help="The task for the agent to solve")
    parser.add_argument("--model", default="openrouter/free", help="Model to use (default: openrouter/free)")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum number of steps (default: 30)")
    args = parser.parse_args()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.task},
    ]

    for step in range(1, args.max_steps + 1):
        print(f"\n--- Step {step}/{args.max_steps} ---")
        try:
            lm_output: str = query_lm(messages, model=args.model)
            print(lm_output)
            messages.append({"role": "assistant", "content": lm_output})
            action: str = parse_action(lm_output)
            print(f"→ {action}")
            output: str = execute_action(action)
            output = truncate_output(output) if output else "(no output)"
            print(output)
            messages.append({"role": "user", "content": output})
        except TerminatingError as e:
            print(f"Done: {e}")
            break
        except subprocess.TimeoutExpired:
            messages.append({"role": "user", "content": (
                "Your last command timed out. You might want to try a "
                "non-interactive alternative or break the task into smaller steps."
            )})
        except Exception as e:
            messages.append({"role": "user", "content": str(e)})
    else:
        print(f"\nReached step limit ({args.max_steps}).")

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trajectory_path = f"logs/trajectory_{ts}.json"
    save_trajectory(messages, trajectory_path)
    print(f"Trajectory saved to {trajectory_path}")


if __name__ == "__main__":
    main()
