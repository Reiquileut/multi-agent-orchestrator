"""Unit tests for individual agents (mocked LLM)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

from src.state import OrchestratorState


def _make_state(**overrides) -> OrchestratorState:
    """Create a minimal valid state for testing."""
    base = {
        "messages": [],
        "task": "Test task: summarize AI trends",
        "plan": ["Research AI trends", "Analyze data", "Write summary"],
        "current_agent": "",
        "agent_outputs": [],
        "iteration": 0,
        "final_output": "",
    }
    base.update(overrides)
    return base


class TestSupervisor:
    """Tests for the Supervisor agent."""

    @pytest.mark.asyncio
    async def test_routes_to_researcher(self):
        """Supervisor should route to researcher when no work has been done."""
        from src.agents.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "researcher\nNeed to gather information first."

        with patch("src.agents.supervisor.get_llm") as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            state = _make_state()
            result = await supervisor_node(state)

            assert result["current_agent"] == "researcher"
            assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_finishes_after_max_iterations(self):
        """Supervisor should FINISH if max iterations reached."""
        from src.agents.supervisor import supervisor_node

        state = _make_state(iteration=10)
        result = await supervisor_node(state)

        assert result["current_agent"] == "FINISH"

    @pytest.mark.asyncio
    async def test_routes_to_writer_after_research(self):
        """Supervisor should route to writer when research is done."""
        from src.agents.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "writer\nResearch is complete, time to write."

        with patch("src.agents.supervisor.get_llm") as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            state = _make_state(
                agent_outputs=[
                    {"agent": "researcher", "output": "Found key data about AI."},
                    {"agent": "analyst", "output": "Key trends: agents, RAG, fine-tuning."},
                ],
                iteration=3,
            )
            result = await supervisor_node(state)

            assert result["current_agent"] == "writer"

    @pytest.mark.asyncio
    async def test_plan_task(self):
        """Planner should produce a list of steps."""
        from src.agents.supervisor import plan_task

        mock_response = MagicMock()
        mock_response.content = "1. Research AI trends\n2. Analyze findings\n3. Write report"

        with patch("src.agents.supervisor.get_llm") as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            state = _make_state()
            result = await plan_task(state)

            assert len(result["plan"]) == 3
            assert result["iteration"] == 0


class TestWriter:
    """Tests for the Writer agent."""

    @pytest.mark.asyncio
    async def test_produces_final_output(self):
        """Writer should set final_output in state."""
        from src.agents.writer import writer_node

        mock_response = MagicMock()
        mock_response.content = "# AI Trends Report\n\nHere are the key findings..."

        with patch("src.agents.writer.get_llm") as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            state = _make_state(
                agent_outputs=[
                    {"agent": "researcher", "output": "Data about AI frameworks."},
                    {"agent": "analyst", "output": "LangGraph leads the market."},
                ],
            )
            result = await writer_node(state)

            assert result["final_output"] is not None
            assert len(result["final_output"]) > 0
            assert result["agent_outputs"][0]["agent"] == "writer"
