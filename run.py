"""tinyagent — a minimal coding agent."""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}


def color(code, text):
    return f"\033[{code}m{text}\033[0m"


class Agent:
    """A minimal coding agent that queries an LLM and executes bash commands."""

    def __init__(self, config: dict, api_key: str) -> None:
        self.client = OpenAI(base_url=config["base_url"], api_key=api_key)
        for k in ("model", "max_steps", "max_context_length", "system_prompt", "instance_template", "compact_prompt", "command_timeout"):
            setattr(self, k, config[k])

    def query(self, messages: list[dict]):
        return (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[BASH_TOOL],
            )
            .choices[0]
            .message
        )

    def compact(self, messages: list[dict]) -> None:
        """Summarize and replace middle messages when approaching the token limit."""
        estimated_tokens = sum(len(m.get("content") or "") for m in messages) // 4
        if estimated_tokens < self.max_context_length * 0.75:
            return
        middle = messages[1:-4]
        if len(middle) < 2:
            return
        middle_text = "\n".join(f"[{m['role']}]: {(m.get('content') or '')[:200]}" for m in middle)
        summary = self.query(
            [
                {"role": "system", "content": self.compact_prompt},
                {"role": "user", "content": middle_text},
            ]
        ).content
        messages[1:-4] = [{"role": "user", "content": f"[Summary of earlier work]\n{summary}"}]
        print(color("1;35", f"\nCompacted {len(middle)} messages ({estimated_tokens} est. tokens)"))

    def run(self, task: str) -> list[dict]:
        """Run the agent loop and return the message history."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.instance_template.replace("{{task}}", task)},
        ]
        for i in range(1, self.max_steps + 1):
            print(color("1;34", f"\n{'=' * 50}\n Step {i}/{self.max_steps}\n{'=' * 50}"))
            self.compact(messages)
            response = self.query(messages)
            messages.append(response.model_dump(exclude_none=True))
            if response.content:
                print(color("1;33", "Response:") + f"\n{response.content}")
            if not response.tool_calls:
                print(color("1;32", "\nDone."))
                break
            command = json.loads(response.tool_calls[0].function.arguments).get("command", "")
            if command == "exit":
                print(color("1;32", "\nDone."))
                break
            print(color("1;36", f"\n$ {command}"))
            try:
                output = (
                    subprocess.run(
                        command,
                        shell=True,
                        text=True,
                        env=os.environ,
                        encoding="utf-8",
                        errors="replace",
                        capture_output=True,
                        timeout=self.command_timeout,
                    ).stdout
                    or "(no output)"
                )
            except subprocess.TimeoutExpired:
                output = color("1;31", "Command timed out.")
            messages.append({"role": "tool", "tool_call_id": response.tool_calls[0].id, "content": output})
            print(f"\n{output}")
        else:
            print(color("1;31", f"\nReached step limit ({self.max_steps})."))
        return messages


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="tinyagent — a minimal coding agent")
    parser.add_argument("task", help="The task for the agent to solve")
    default_config = Path(__file__).parent / "config" / "default.yaml"
    parser.add_argument("--config", default=str(default_config), help="Path to YAML config file")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--base-url", help="Override API base URL")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--max-context-length", type=int, help="Override max context length")
    parser.add_argument("--command-timeout", type=int, help="Override command timeout in seconds")
    args = parser.parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)
    for key in ("model", "base_url", "max_steps", "max_context_length", "command_timeout"):
        if (val := getattr(args, key, None)) is not None:
            config[key] = val

    agent = Agent(config=config, api_key=os.getenv("TINYAGENT_API_KEY"))
    messages = agent.run(args.task)

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = f"logs/trajectory_{ts}.json"
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"Trajectory saved to {path}")


if __name__ == "__main__":
    main()
