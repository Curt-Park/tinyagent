"""Tests for lace.__main__."""

import json

import pytest

from run import TerminatingError, parse_action, save_trajectory, truncate_output


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


class TestSaveTrajectory:
    def test_writes_json(self, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        path = str(tmp_path / "traj.json")
        save_trajectory(messages, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == messages
