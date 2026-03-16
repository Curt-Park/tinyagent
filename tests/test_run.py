"""Tests for tinyagent."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run import Agent, load_config

TEST_CONFIG = {
    "system_prompt": "test prompt",
    "instance_template": "{{task}}",
    "compact_prompt": "test compact",
    "model": "test",
    "base_url": "http://test",
    "max_steps": 30,
    "max_context_length": 16000,
    "command_timeout": 30,
}


def _make_tool_response(command: str, tool_call_id: str = "call_1"):
    """Create a mock response with a tool call."""
    tc = SimpleNamespace(id=tool_call_id, type="function", function=SimpleNamespace(name="bash", arguments=json.dumps({"command": command})))
    return SimpleNamespace(
        content=None,
        tool_calls=[tc],
        model_dump=lambda exclude_none=False: {"role": "assistant", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": "bash", "arguments": tc.function.arguments}}]},
    )


def _make_text_response(text: str):
    """Create a mock response with text only (no tool calls)."""
    return SimpleNamespace(content=text, tool_calls=None, model_dump=lambda exclude_none=False: {"role": "assistant", "content": text})


@pytest.fixture
def agent():
    with patch("run.OpenAI"):
        return Agent(config=TEST_CONFIG, api_key="test")


class TestRun:
    def test_exit_command_stops_loop(self, agent):
        agent.client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=_make_tool_response("exit"))])
        messages = agent.run("do something")
        assert len(messages) == 3  # system + user + assistant

    @patch("run.subprocess.run")
    def test_tool_call_executes_command(self, mock_subprocess, agent):
        mock_subprocess.return_value = MagicMock(stdout="file1.txt\nfile2.txt\n")
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=_make_tool_response("ls", "call_1"))]),
            SimpleNamespace(choices=[SimpleNamespace(message=_make_tool_response("exit", "call_2"))]),
        ]
        agent.client.chat.completions.create.side_effect = responses
        messages = agent.run("list files")
        mock_subprocess.assert_called_once()
        # system + user + assistant(ls) + tool(output) + assistant(exit)
        assert len(messages) == 5

    def test_no_tool_calls_stops_loop(self, agent):
        agent.client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=_make_text_response("All done!"))])
        messages = agent.run("do something")
        assert len(messages) == 3
        assert messages[-1]["content"] == "All done!"


class TestCompact:
    def _make_messages(self, middle_count: int, char_per_msg: int = 100) -> list:
        msgs = [{"role": "system", "content": "system prompt"}]
        for i in range(middle_count):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": "x" * char_per_msg})
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"recent_{i}"})
        return msgs

    def test_no_compaction_below_threshold(self, agent):
        msgs = self._make_messages(4, char_per_msg=10)
        original = [m.copy() for m in msgs]
        agent.compact(msgs)
        assert msgs == original

    @patch.object(Agent, "query", return_value=SimpleNamespace(content="Summary of work done."))
    def test_compaction_replaces_middle(self, mock_query, agent):
        msgs = self._make_messages(10, char_per_msg=400)
        original_len = len(msgs)
        agent.max_context_length = 1000
        agent.compact(msgs)
        assert len(msgs) < original_len
        assert msgs[1]["content"].startswith("[Summary of earlier work]")
        mock_query.assert_called_once()

    @patch.object(Agent, "query", return_value=SimpleNamespace(content="Summary."))
    def test_preserves_system_and_recent(self, mock_query, agent):
        msgs = self._make_messages(10, char_per_msg=400)
        system = msgs[0].copy()
        recent_4 = [m.copy() for m in msgs[-4:]]
        agent.max_context_length = 1000
        agent.compact(msgs)
        assert msgs[0] == system
        assert msgs[-4:] == recent_4

    def test_skips_when_middle_too_small(self, agent):
        msgs = [
            {"role": "system", "content": "s" * 4000},
            {"role": "user", "content": "m" * 4000},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "r2"},
            {"role": "user", "content": "r3"},
            {"role": "assistant", "content": "r4"},
        ]
        original = [m.copy() for m in msgs]
        agent.max_context_length = 1000
        agent.compact(msgs)
        assert msgs == original

    def test_handles_none_content(self, agent):
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "x" * 100},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "r2"},
            {"role": "user", "content": "r3"},
            {"role": "assistant", "content": "r4"},
        ]
        # Should not raise on None content
        agent.compact(msgs)


class TestConfig:
    def test_loads_default_config(self):
        config = load_config("config/default.yaml")
        assert config["model"] == "openrouter/free"
        assert config["max_steps"] == 30
        assert config["command_timeout"] == 30

    def test_loads_custom_config(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model: gpt-4o\nmax_steps: 10\ncommand_timeout: 60\n")
        config = load_config(str(cfg_file))
        assert config["model"] == "gpt-4o"
        assert config["max_steps"] == 10
        assert config["command_timeout"] == 60


class TestSaveTrajectory:
    def test_writes_json(self, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        path = str(tmp_path / "traj.json")
        with open(path, "w") as f:
            json.dump(messages, f, indent=2)
        with open(path) as f:
            assert json.load(f) == messages
