"""Tests for tinyagent."""

import json
from unittest.mock import patch

import pytest

from run import Agent


@pytest.fixture
def agent():
    with patch("run.OpenAI"):
        return Agent(model="test", base_url="http://test", api_key="test")


class TestParseAction:
    def test_extracts_command(self):
        assert Agent.parse_action("THOUGHT: list\n```bash\nls -la\n```") == "ls -la"

    def test_extracts_bash_action(self):
        assert Agent.parse_action("THOUGHT: list\n```bash-action\nls -la\n```") == "ls -la"

    def test_exit_returns_none(self):
        assert Agent.parse_action("THOUGHT: done\n```bash\nexit\n```") is None

    def test_no_block_returns_none(self):
        assert Agent.parse_action("I will just say hello") is None


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

    @patch.object(Agent, "query", return_value="Summary of work done.")
    def test_compaction_replaces_middle(self, mock_query, agent):
        msgs = self._make_messages(10, char_per_msg=400)
        original_len = len(msgs)
        agent.max_context_length = 1000
        agent.compact(msgs)
        assert len(msgs) < original_len
        assert msgs[1]["content"].startswith("[Summary of earlier work]")
        mock_query.assert_called_once()

    @patch.object(Agent, "query", return_value="Summary.")
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


class TestSaveTrajectory:
    def test_writes_json(self, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        path = str(tmp_path / "traj.json")
        with open(path, "w") as f:
            json.dump(messages, f, indent=2)
        with open(path) as f:
            assert json.load(f) == messages
