"""tinyagent — a minimal coding agent."""

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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

COMPACT_PROMPT = "Summarize the following agent conversation history in one concise paragraph. Focus on: what was done, what was learned, and what remains."


class Agent:
    """A minimal coding agent that queries an LLM and executes bash commands."""

    def __init__(self, model: str, base_url: str, api_key: str, max_steps: int = 30, max_context_length: int = 16000) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_steps = max_steps
        self.max_context_length = max_context_length

    def query(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the language model and return the response text."""
        return self.client.chat.completions.create(model=self.model, messages=messages).choices[0].message.content

    @staticmethod
    def parse_action(text: str) -> str | None:
        """Extract a bash command from a fenced code block. Return None to stop."""
        matches = re.findall(r"```(?:bash)\s*\n?(.*?)\n?```", text, re.DOTALL)
        if not matches or matches[0].strip() == "exit":
            return None
        return matches[0].strip()

    def compact(self, messages: list[dict[str, str]]) -> None:
        """Summarize and replace middle messages when approaching the token limit."""
        estimated_tokens = sum(len(m["content"]) for m in messages) // 4
        if estimated_tokens < self.max_context_length * 0.75:
            return

        middle = messages[1:-4]
        if len(middle) < 2:
            return

        middle_text = "\n".join(f"[{m['role']}]: {m['content'][:200]}" for m in middle)
        summary = self.query([
            {"role": "system", "content": COMPACT_PROMPT},
            {"role": "user", "content": middle_text},
        ])
        messages[1:-4] = [{"role": "user", "content": f"[Summary of earlier work]\n{summary}"}]
        print(f"[compact] Replaced {len(middle)} messages with summary ({estimated_tokens} est. tokens)")

    def run(self, task: str) -> list[dict[str, str]]:
        """Run the agent loop and return the message history."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        for i in range(1, self.max_steps + 1):
            print(f"\n--- Step {i}/{self.max_steps} ---")
            try:
                self.compact(messages)
                response = self.query(messages)
                print(response)
                messages.append({"role": "assistant", "content": response})

                action = self.parse_action(response)
                if action is None:
                    print("Done.")
                    break

                print(f"→ {action}")
                result = subprocess.run(
                    action, shell=True, text=True, env=os.environ,
                    encoding="utf-8", errors="replace",
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
                )
                messages.append({"role": "user", "content": result.stdout or "(no output)"})
                print(messages[-1]["content"])
            except subprocess.TimeoutExpired:
                messages.append({"role": "user", "content": "Command timed out."})
            except Exception as e:
                messages.append({"role": "user", "content": str(e)})
        else:
            print(f"\nReached step limit ({self.max_steps}).")
        return messages


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="tinyagent — a minimal coding agent")
    parser.add_argument("task", help="The task for the agent to solve")
    parser.add_argument("--model", default="openrouter/free", help="Model to use")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="LLM API base URL")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum number of steps")
    parser.add_argument("--max-context-length", type=int, default=16000, help="Estimated token budget")
    args = parser.parse_args()

    agent = Agent(
        model=args.model, base_url=args.base_url, api_key=os.getenv("TINYAGENT_API_KEY"),
        max_steps=args.max_steps, max_context_length=args.max_context_length,
    )
    messages = agent.run(args.task)

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = f"logs/trajectory_{ts}.json"
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"Trajectory saved to {path}")


if __name__ == "__main__":
    main()
