import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _tool(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _tool("bash", "Execute a bash command", {"command": {"type": "string"}}, ["command"]),
    _tool("web_search", "Search the web", {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}}, ["query"]),
]


class Agent:
    def __init__(self, cfg: dict[str, Any], api_key: str) -> None:
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
        for k in ("model", "max_steps", "max_context_length", "system_prompt", "instance_template", "compact_prompt", "command_timeout"):
            setattr(self, k, cfg[k])

    def query(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS).choices[0].message

    def compact(self, messages: list[dict[str, Any]]) -> None:
        est = sum(len(m.get("content") or "") for m in messages) // 4
        if est < self.max_context_length * 0.75 or len(messages[1:-4]) < 2:
            return
        middle = "\n".join(f"[{m['role']}]: {(m.get('content') or '')[:200]}" for m in messages[1:-4])
        summary = self.query([{"role": "system", "content": self.compact_prompt}, {"role": "user", "content": middle}]).content
        messages[1:-4] = [{"role": "user", "content": f"[Summary]\n{summary}"}]
        print(f"\033[1;35mCompacted {len(messages[1:-4])} messages ({est} est. tokens)\033[0m")

    def _exec(self, name: str, args: dict[str, Any]) -> str | None:
        if name == "web_search":
            print(f"\033[1;36m🔍 {args['query']}\033[0m")
            results = DDGS().text(args["query"], max_results=args.get("max_results", 5))
            return "\n\n".join(f"[{r['title']}]({r['href']})\n{r['body']}" for r in results) or "(no results)"
        cmd = args.get("command", "")
        if not cmd or cmd == "exit":
            return None
        print(f"\033[1;36m$ {cmd}\033[0m")
        try:
            return subprocess.run(cmd, shell=True, text=True, env=os.environ, encoding="utf-8", errors="replace", capture_output=True, timeout=self.command_timeout).stdout or "(no output)"
        except subprocess.TimeoutExpired:
            return "\033[1;31mCommand timed out.\033[0m"

    def run(self, task: str) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.instance_template.replace("{{task}}", task)}]
        for i in range(1, self.max_steps + 1):
            print(f"\033[1;34m\n{'=' * 50}\n Step {i}/{self.max_steps}\n{'=' * 50}\033[0m")
            self.compact(messages)
            resp = self.query(messages)
            messages.append(resp.model_dump(exclude_none=True))
            if resp.content:
                print(f"\033[1;33mResponse:\033[0m\n{resp.content}")
            tc = (resp.tool_calls or [None])[0]
            if not tc:
                break
            output = self._exec(tc.function.name, json.loads(tc.function.arguments))
            if output is None:
                break
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            print(output)
        else:
            print(f"\033[1;31mReached step limit ({self.max_steps}).\033[0m")
        return messages


def main() -> None:
    overrides = {"model": {}, "base_url": {}, "max_steps": {"type": int}, "max_context_length": {"type": int}, "command_timeout": {"type": int}}
    parser = argparse.ArgumentParser(description="tinyagent — a minimal coding agent")
    parser.add_argument("task", help="The task for the agent to solve")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    for key, kw in overrides.items():
        parser.add_argument(f"--{key.replace('_', '-')}", **kw)
    args = parser.parse_args()
    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k in overrides if (v := getattr(args, k, None)) is not None})
    messages = Agent(cfg, os.getenv("TINYAGENT_API_KEY")).run(args.task)
    log = Path("logs") / f"trajectory_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    log.parent.mkdir(exist_ok=True)
    log.write_text(json.dumps(messages, indent=2))
    print(f"Trajectory saved to {log}")


if __name__ == "__main__":
    main()
