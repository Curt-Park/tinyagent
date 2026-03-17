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


class Agent:
    def __init__(self, cfg, api_key):
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
        for k in ("model", "max_steps", "max_context_length", "system_prompt", "instance_template", "compact_prompt", "command_timeout"):
            setattr(self, k, cfg[k])

    def query(self, messages):
        tool = {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]},
            },
        }
        return self.client.chat.completions.create(model=self.model, messages=messages, tools=[tool]).choices[0].message

    def compact(self, messages):
        est = sum(len(m.get("content") or "") for m in messages) // 4
        if est < self.max_context_length * 0.75 or len(messages[1:-4]) < 2:
            return
        middle = "\n".join(f"[{m['role']}]: {(m.get('content') or '')[:200]}" for m in messages[1:-4])
        summary = self.query([{"role": "system", "content": self.compact_prompt}, {"role": "user", "content": middle}]).content
        messages[1:-4] = [{"role": "user", "content": f"[Summary]\n{summary}"}]
        print(f"\033[1;35mCompacted {len(messages[1:-4])} messages ({est} est. tokens)\033[0m")

    def run(self, task):
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.instance_template.replace("{{task}}", task)}]
        for i in range(1, self.max_steps + 1):
            print(f"\033[1;34m\n{'=' * 50}\n Step {i}/{self.max_steps}\n{'=' * 50}\033[0m")
            self.compact(messages)
            resp = self.query(messages)
            messages.append(resp.model_dump(exclude_none=True))
            if resp.content:
                print(f"\033[1;33mResponse:\033[0m\n{resp.content}")
            tool = (resp.tool_calls or [None])[0]
            if not tool:
                break
            cmd = json.loads(tool.function.arguments).get("command", "")
            if not cmd or cmd == "exit":
                break
            print(f"\033[1;36m$ {cmd}\033[0m")
            try:
                output = (
                    subprocess.run(cmd, shell=True, text=True, env=os.environ, encoding="utf-8", errors="replace", capture_output=True, timeout=self.command_timeout).stdout
                    or "(no output)"
                )
            except subprocess.TimeoutExpired:
                output = "\033[1;31mCommand timed out.\033[0m"
            messages.append({"role": "tool", "tool_call_id": tool.id, "content": output})
            print(output)
        else:
            print(f"\033[1;31mReached step limit ({self.max_steps}).\033[0m")
        return messages


CONFIG_OVERRIDES = {"model": {}, "base_url": {}, "max_steps": {"type": int}, "max_context_length": {"type": int}, "command_timeout": {"type": int}}


def main():
    parser = argparse.ArgumentParser(description="tinyagent — a minimal coding agent")
    parser.add_argument("task", help="The task for the agent to solve")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    for key, kw in CONFIG_OVERRIDES.items():
        parser.add_argument(f"--{key.replace('_', '-')}", **kw)
    args = parser.parse_args()
    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)
    for k in CONFIG_OVERRIDES:
        if (v := getattr(args, k, None)) is not None:
            cfg[k] = v
    messages = Agent(cfg, os.getenv("TINYAGENT_API_KEY")).run(args.task)
    os.makedirs("logs", exist_ok=True)
    path = f"logs/trajectory_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"Trajectory saved to {path}")


if __name__ == "__main__":
    main()
