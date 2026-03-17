"""Tests for tinyagent."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from run import Agent, main

CFG = {
    "system_prompt": "test prompt",
    "instance_template": "{{task}}",
    "compact_prompt": "test compact",
    "model": "test",
    "base_url": "http://test",
    "max_steps": 30,
    "max_context_length": 16000,
    "command_timeout": 30,
}


def _resp(name, args, tool_call_id="call_1"):
    tc = SimpleNamespace(id=tool_call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    msg = SimpleNamespace(
        content=None,
        tool_calls=[tc],
        model_dump=lambda exclude_none=False: {"role": "assistant", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _text_resp(text):
    msg = SimpleNamespace(content=text, tool_calls=None, model_dump=lambda exclude_none=False: {"role": "assistant", "content": text})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _bash(cmd, call_id="call_1"):
    return _resp("bash", {"command": cmd}, call_id)


def _search(query, call_id="call_1"):
    return _resp("web_search", {"query": query}, call_id)


@pytest.fixture
def agent():
    with patch("run.OpenAI"):
        return Agent(CFG, "test")


# --- run loop ---


class TestRun:
    def test_exit_stops(self, agent):
        agent.client.chat.completions.create.return_value = _bash("exit")
        assert len(agent.run("task")) == 3  # system + user + assistant

    def test_no_tool_calls_stops(self, agent):
        agent.client.chat.completions.create.return_value = _text_resp("Done!")
        msgs = agent.run("task")
        assert len(msgs) == 3
        assert msgs[-1]["content"] == "Done!"

    @patch("run.subprocess.run")
    def test_tool_call_executes(self, mock_sub, agent):
        mock_sub.return_value = MagicMock(stdout="output\n")
        agent.client.chat.completions.create.side_effect = [_bash("ls", "c1"), _bash("exit", "c2")]
        msgs = agent.run("list files")
        mock_sub.assert_called_once()
        assert len(msgs) == 5  # system + user + assistant + tool + assistant

    def test_step_limit(self, agent):
        agent.max_steps = 2
        agent.client.chat.completions.create.side_effect = [_bash("echo 1", "c1"), _bash("echo 2", "c2")]
        with patch("run.subprocess.run", return_value=MagicMock(stdout="ok")):
            assert len(agent.run("loop")) == 6  # system + user + 2*(assistant + tool)

    @patch("run.DDGS")
    def test_web_search_in_loop(self, mock_ddgs, agent):
        mock_ddgs.return_value.text.return_value = [{"title": "Doc", "href": "http://doc.com", "body": "Info"}]
        agent.client.chat.completions.create.side_effect = [_search("python"), _text_resp("Done!")]
        msgs = agent.run("search")
        assert any("Doc" in (m.get("content") or "") for m in msgs)


# --- _exec ---


class TestExec:
    @patch("run.DDGS")
    def test_web_search(self, mock_ddgs, agent):
        mock_ddgs.return_value.text.return_value = [{"title": "R", "href": "http://r.com", "body": "body"}]
        output = agent._exec_web_search({"query": "q"})
        assert "R" in output and "http://r.com" in output
        mock_ddgs.return_value.text.assert_called_once_with("q", max_results=5)

    @patch("run.DDGS")
    def test_web_search_no_results(self, mock_ddgs, agent):
        mock_ddgs.return_value.text.return_value = []
        assert agent._exec_web_search({"query": "q"}) == "(no results)"

    @patch("run.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1))
    def test_bash_timeout(self, _mock, agent):
        assert "timed out" in agent._exec_bash({"command": "sleep 999"})

    @pytest.mark.parametrize("cmd", ["exit", ""])
    def test_bash_returns_none(self, agent, cmd):
        assert agent._exec_bash({"command": cmd}) is None


# --- compact ---


class TestCompact:
    def _msgs(self, n, size=100):
        msgs = [{"role": "system", "content": "sys"}]
        msgs += [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * size} for i in range(n)]
        msgs += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"r{i}"} for i in range(4)]
        return msgs

    def test_no_compaction_below_threshold(self, agent):
        msgs = self._msgs(4, size=10)
        original = [m.copy() for m in msgs]
        agent.compact(msgs)
        assert msgs == original

    @patch.object(Agent, "query", return_value=SimpleNamespace(content="Summary."))
    def test_compaction_replaces_middle(self, _mock, agent):
        msgs = self._msgs(10, size=400)
        system, recent = msgs[0].copy(), [m.copy() for m in msgs[-4:]]
        agent.max_context_length = 1000
        agent.compact(msgs)
        assert msgs[1]["content"].startswith("[Summary]")
        assert msgs[0] == system
        assert msgs[-4:] == recent

    def test_skips_when_middle_too_small(self, agent):
        msgs = [{"role": "system", "content": "s" * 4000}, {"role": "user", "content": "m" * 4000}]
        msgs += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"r{i}"} for i in range(4)]
        original = [m.copy() for m in msgs]
        agent.max_context_length = 1000
        agent.compact(msgs)
        assert msgs == original

    def test_handles_none_content(self, agent):
        msgs = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": None}, {"role": "user", "content": "x" * 100}]
        msgs += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"r{i}"} for i in range(4)]
        agent.compact(msgs)  # should not raise


# --- main ---


class TestMain:
    @pytest.fixture
    def cfg_file(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump(CFG))
        return str(f)

    @patch("run.Agent")
    def test_main_runs(self, mock_cls, tmp_path, cfg_file, monkeypatch):
        mock_cls.return_value.run.return_value = [{"role": "user", "content": "hi"}]
        monkeypatch.setattr("sys.argv", ["tinyagent", "task", "--config", cfg_file])
        monkeypatch.chdir(tmp_path)
        main()
        mock_cls.return_value.run.assert_called_once_with("task")
        logs = list(tmp_path.glob("logs/trajectory_*.json"))
        assert len(logs) == 1
        assert json.loads(logs[0].read_text()) == [{"role": "user", "content": "hi"}]

    @patch("run.Agent")
    def test_main_overrides(self, mock_cls, tmp_path, cfg_file, monkeypatch):
        mock_cls.return_value.run.return_value = []
        monkeypatch.setattr("sys.argv", ["tinyagent", "task", "--config", cfg_file, "--max-steps", "5"])
        monkeypatch.chdir(tmp_path)
        main()
        assert mock_cls.call_args[0][0]["max_steps"] == 5
