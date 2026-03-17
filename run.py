import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BLUE, CYAN, MAGENTA, RED, YELLOW, RESET = (f"\033[1;{c}m" for c in (34, 36, 35, 31, 33, 0))
TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": r}}}
    for n, d, p, r in [
        ("bash", "Execute a bash command", {"command": {"type": "string"}}, ["command"]),
        ("web_search", "Search the web", {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}}, ["query"]),
    ]
]


class Agent:
    def __init__(self, cfg: dict, api_key: str) -> None:
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
        for key in ("model", "max_steps", "max_context_length", "system_prompt", "instance_template", "compact_prompt", "command_timeout"):
            setattr(self, key, cfg[key])

    def query(self, messages: list[dict]) -> object:
        return self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS).choices[0].message

    def compact(self, messages: list[dict]) -> None:
        est_tokens = sum(len(m.get("content") or "") for m in messages) // 4
        if est_tokens < self.max_context_length * 0.75 or len(messages[1:-4]) < 2:
            return
        middle = "\n".join(f"[{m['role']}]: {(m.get('content') or '')[:200]}" for m in messages[1:-4])
        summary = self.query([{"role": "system", "content": self.compact_prompt}, {"role": "user", "content": middle}]).content
        messages[1:-4] = [{"role": "user", "content": f"[Summary]\n{summary}"}]
        print(f"{MAGENTA}Compacted {len(messages[1:-4])} messages ({est_tokens} est. tokens){RESET}")

    def _exec_bash(self, args: dict) -> str | None:
        cmd = args.get("command", "")
        if not cmd or cmd == "exit":
            return None
        print(f"{CYAN}$ {cmd}{RESET}")
        try:
            result = subprocess.run(cmd, shell=True, text=True, env=os.environ, encoding="utf-8", errors="replace", capture_output=True, timeout=self.command_timeout)
            return result.stdout or "(no output)"
        except subprocess.TimeoutExpired:
            return f"{RED}Command timed out.{RESET}"

    def _exec_web_search(self, args: dict) -> str:
        print(f"{CYAN}search:{args['query']}{RESET}")
        return "\n\n".join(f"[{r['title']}]({r['href']})\n{r['body']}" for r in DDGS().text(args["query"], max_results=args.get("max_results", 5))) or "(no results)"

    def run(self, task: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.instance_template.replace("{{task}}", task)}]
        for step in range(1, self.max_steps + 1):
            print(f"{BLUE}\n{'=' * 50}\n Step {step}/{self.max_steps}\n{'=' * 50}{RESET}")
            self.compact(messages)
            response = self.query(messages)
            messages.append(response.model_dump(exclude_none=True))
            if response.content:
                print(f"{YELLOW}Response:{RESET}\n{response.content}")
            tool_call = (response.tool_calls or [None])[0]
            if not tool_call:
                break
            output = {"bash": self._exec_bash, "web_search": self._exec_web_search}[tool_call.function.name](json.loads(tool_call.function.arguments))
            if output is None:
                break
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
            print(output)
        else:
            print(f"{RED}Reached step limit ({self.max_steps}).{RESET}")
        return messages


def main() -> None:
    overrides = {"model": {}, "base_url": {}, "max_steps": {"type": int}, "max_context_length": {"type": int}, "command_timeout": {"type": int}}
    parser = argparse.ArgumentParser(description="tinyagent — a minimal coding agent")
    parser.add_argument("task", help="The task for the agent to solve")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "default.yaml"))
    for key, kwargs in overrides.items():
        parser.add_argument(f"--{key.replace('_', '-')}", **kwargs)
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
