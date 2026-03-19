"""Unit tests for the CLI entry point."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestRunTask:
    """Tests for the run_task function."""

    @pytest.mark.asyncio
    async def test_run_task_non_verbose(self):
        """run_task should invoke orchestrator and return final output."""
        mock_state = MagicMock()
        mock_state.values = {"final_output": "Task completed."}

        with patch("src.cli.orchestrator") as mock_orch:
            mock_orch.ainvoke = AsyncMock(return_value={})
            mock_orch.aget_state = AsyncMock(return_value=mock_state)

            from src.cli import run_task

            result = await run_task("test task", verbose=False)

            assert result == "Task completed."
            mock_orch.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_task_verbose(self):
        """run_task in verbose mode should stream events without error."""
        mock_state = MagicMock()
        mock_state.values = {"final_output": "Verbose output."}

        async def fake_stream(*args, **kwargs):
            yield {
                "event": "on_chain_end",
                "name": "researcher",
                "data": {
                    "output": {
                        "agent_outputs": [{"output": "data"}],
                        "current_agent": "analyst",
                    }
                },
            }

        with patch("src.cli.orchestrator") as mock_orch:
            mock_orch.astream_events = fake_stream
            mock_orch.aget_state = AsyncMock(return_value=mock_state)

            from src.cli import run_task

            result = await run_task("test task", verbose=True)

            assert result == "Verbose output."


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_main_with_positional_arg(self):
        """main() should call run_task with positional argument."""
        with patch("src.cli.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                task="hello", task_flag=None, verbose=False
            )

            with patch("src.cli.asyncio.run") as mock_run:
                from src.cli import main

                main()
                mock_run.assert_called_once()

    def test_main_with_flag_arg(self):
        """main() should work with --task flag."""
        with patch("src.cli.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                task=None, task_flag="foo", verbose=False
            )

            with patch("src.cli.asyncio.run") as mock_run:
                from src.cli import main

                main()
                mock_run.assert_called_once()

    def test_main_no_task_exits(self):
        """main() should error when no task is provided."""
        with patch("src.cli.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                task=None, task_flag=None, verbose=False
            )

            with patch(
                "src.cli.argparse.ArgumentParser.error", side_effect=SystemExit(2)
            ):
                from src.cli import main

                with pytest.raises(SystemExit):
                    main()

    def test_main_module_entry(self):
        """The __name__ == '__main__' block should exist."""
        import ast

        with open("src/cli.py") as f:
            tree = ast.parse(f.read())

        has_main_block = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(
                isinstance(c, ast.Constant) and c.value == "__main__"
                for c in node.test.comparators
            )
            for node in ast.walk(tree)
        )
        assert has_main_block
