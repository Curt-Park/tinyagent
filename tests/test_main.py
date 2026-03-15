"""Tests for lace.__main__."""

import json
from unittest.mock import patch

import pytest

from run import (
    TerminatingError,
    compact_messages,
    parse_action,
    save_trajectory,
    truncate_output,
)


class TestParseAction:
    def test_extracts_command(self):
        output = "THOUGHT: list files\n```bash\nls -la\n```"
        assert parse_action(output) == "ls -la"

    def test_extracts_command_bash_action(self):
        output = "THOUGHT: list files\n```bash-action\nls -la\n```"
        assert parse_action(output) == "ls -la"

    def test_exit_raises(self):
        output = "THOUGHT: done\n```bash\nexit\n```"
        with pytest.raises(TerminatingError, match="exit"):
            parse_action(output)

    def test_no_block_raises(self):
        output = "I will just say hello"
        with pytest.raises(TerminatingError, match="No action block found"):
            parse_action(output)


class TestTruncateOutput:
    def test_short(self):
        assert truncate_output("hello", max_chars=100) == "hello"

    def test_long(self):
        text = "A" * 20_000
        result = truncate_output(text, max_chars=10_000)
        assert "[...truncated 10000 chars...]" in result
        assert len(result) < len(text)
        assert result.startswith("A" * 5000)
        assert result.endswith("A" * 5000)

    def test_empty(self):
        assert truncate_output("") == ""


class TestCompactMessages:
    def _make_messages(self, middle_count: int, char_per_msg: int = 100) -> list:
        msgs = [{"role": "system", "content": "system prompt"}]
        for i in range(middle_count):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": "x" * char_per_msg})
        # Last 4 messages (2 turn pairs)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"recent_{i}"})
        return msgs

    def test_no_compaction_below_threshold(self):
        msgs = self._make_messages(4, char_per_msg=10)
        original = [m.copy() for m in msgs]
        compact_messages(msgs, "test-model", max_context_tokens=16000)
        assert msgs == original

    @patch("run.query_lm", return_value="Summary of work done.")
    def test_compaction_replaces_middle(self, mock_query_lm):
        # 10 middle msgs * 400 chars = 4000 chars => ~1000 tokens
        # + system + 4 recent ≈ need threshold low enough to trigger
        msgs = self._make_messages(10, char_per_msg=400)
        original_len = len(msgs)
        # Set max so 75% threshold is exceeded: total ~4400 chars / 4 = ~1100 tokens
        compact_messages(msgs, "test-model", max_context_tokens=1000)
        assert len(msgs) < original_len
        assert msgs[1]["content"].startswith("[Summary of earlier work]")
        mock_query_lm.assert_called_once()

    @patch("run.query_lm", return_value="Summary.")
    def test_preserves_system_and_recent(self, mock_query_lm):
        msgs = self._make_messages(10, char_per_msg=400)
        system = msgs[0].copy()
        recent_4 = [m.copy() for m in msgs[-4:]]
        compact_messages(msgs, "test-model", max_context_tokens=1000)
        assert msgs[0] == system
        assert msgs[-4:] == recent_4

    def test_skips_when_middle_too_small(self):
        # Only system + 4 recent + 1 middle = middle has < 2 messages
        msgs = [
            {"role": "system", "content": "s" * 4000},
            {"role": "user", "content": "m" * 4000},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "r2"},
            {"role": "user", "content": "r3"},
            {"role": "assistant", "content": "r4"},
        ]
        original = [m.copy() for m in msgs]
        compact_messages(msgs, "test-model", max_context_tokens=1000)
        assert msgs == original


class TestSaveTrajectory:
    def test_writes_json(self, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        path = str(tmp_path / "traj.json")
        save_trajectory(messages, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == messages
