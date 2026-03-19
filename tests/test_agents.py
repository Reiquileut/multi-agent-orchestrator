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


    @pytest.mark.asyncio
    async def test_finishes_when_writer_has_output(self):
        """Supervisor should FINISH when writer has already produced output."""
        from src.agents.supervisor import supervisor_node

        state = _make_state(
            agent_outputs=[
                {"agent": "researcher", "output": "Research done."},
                {"agent": "analyst", "output": "Analysis done."},
                {"agent": "writer", "output": "Final report."},
            ],
            final_output="Final report.",
            iteration=3,
        )
        result = await supervisor_node(state)
        assert result["current_agent"] == "FINISH"

    @pytest.mark.asyncio
    async def test_enforces_pipeline_order_analyst(self):
        """Supervisor should route to analyst when researcher done but analyst not."""
        from src.agents.supervisor import supervisor_node

        state = _make_state(
            agent_outputs=[
                {"agent": "researcher", "output": "Research done."},
            ],
            iteration=1,
        )
        result = await supervisor_node(state)
        assert result["current_agent"] == "analyst"

    @pytest.mark.asyncio
    async def test_enforces_pipeline_order_writer(self):
        """Supervisor should route to writer when researcher+analyst done but writer not."""
        from src.agents.supervisor import supervisor_node

        state = _make_state(
            agent_outputs=[
                {"agent": "researcher", "output": "Research done."},
                {"agent": "analyst", "output": "Analysis done."},
            ],
            iteration=2,
        )
        result = await supervisor_node(state)
        assert result["current_agent"] == "writer"

    @pytest.mark.asyncio
    async def test_blocks_exhausted_agent(self):
        """Supervisor should auto-advance when LLM picks an exhausted agent."""
        from src.agents.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "researcher\nNeed more research."

        with patch("src.agents.supervisor.get_llm") as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            state = _make_state(
                agent_outputs=[
                    {"agent": "researcher", "output": "R1"},
                    {"agent": "researcher", "output": "R2"},
                    {"agent": "analyst", "output": "A1"},
                    {"agent": "analyst", "output": "A2"},
                    {"agent": "writer", "output": "W1"},
                ],
                iteration=5,
            )
            result = await supervisor_node(state)
            # researcher is exhausted (2 calls), should auto-advance
            assert result["current_agent"] != "researcher"

    @pytest.mark.asyncio
    async def test_finish_when_all_exhausted(self):
        """Supervisor should FINISH when all agents have hit their call limits."""
        from src.agents.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "researcher\nNeed more data."

        with patch("src.agents.supervisor.get_llm") as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            state = _make_state(
                agent_outputs=[
                    {"agent": "researcher", "output": "R1"},
                    {"agent": "researcher", "output": "R2"},
                    {"agent": "analyst", "output": "A1"},
                    {"agent": "analyst", "output": "A2"},
                    {"agent": "writer", "output": "W1"},
                    {"agent": "writer", "output": "W2"},
                ],
                iteration=5,
            )
            result = await supervisor_node(state)
            assert result["current_agent"] == "FINISH"

    @pytest.mark.asyncio
    async def test_plan_task_fallback(self):
        """Planner should use default plan when LLM returns no numbered steps."""
        from src.agents.supervisor import plan_task

        mock_response = MagicMock()
        mock_response.content = "Just do the thing without any numbers."

        with patch("src.agents.supervisor.get_llm") as mock_llm:
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            state = _make_state()
            result = await plan_task(state)

            assert result["plan"] == ["Research the topic", "Analyze findings", "Write the output"]


class TestResearcher:
    """Tests for the Researcher agent."""

    @pytest.mark.asyncio
    async def test_researcher_node_returns_output(self):
        """Researcher should return agent_outputs and messages."""
        from src.agents.researcher import researcher_node

        mock_result = {"messages": [MagicMock(content="Found relevant AI data.")]}

        with patch("src.agents.researcher.create_react_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_agent

            with patch("src.agents.researcher.get_llm"):
                state = _make_state()
                result = await researcher_node(state)

                assert result["agent_outputs"][0]["agent"] == "researcher"
                assert result["agent_outputs"][0]["output"] == "Found relevant AI data."
                assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_researcher_formats_previous_findings(self):
        """Researcher should format previous outputs into context."""
        from src.agents.researcher import researcher_node

        mock_result = {"messages": [MagicMock(content="More data found.")]}

        with patch("src.agents.researcher.create_react_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_agent

            with patch("src.agents.researcher.get_llm"):
                state = _make_state(
                    agent_outputs=[
                        {"agent": "researcher", "output": "First round of research."},
                    ],
                )
                result = await researcher_node(state)

                assert result["agent_outputs"][0]["agent"] == "researcher"

    @pytest.mark.asyncio
    async def test_researcher_handles_empty_messages(self):
        """Researcher should return 'No findings.' when messages are empty."""
        from src.agents.researcher import researcher_node

        mock_result = {"messages": []}

        with patch("src.agents.researcher.create_react_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_agent

            with patch("src.agents.researcher.get_llm"):
                state = _make_state()
                result = await researcher_node(state)

                assert result["agent_outputs"][0]["output"] == "No findings."


class TestAnalyst:
    """Tests for the Analyst agent."""

    @pytest.mark.asyncio
    async def test_analyst_node_returns_output(self):
        """Analyst should return agent_outputs and messages."""
        from src.agents.analyst import analyst_node

        mock_result = {"messages": [MagicMock(content="Key insight: AI is growing.")]}

        with patch("langgraph.prebuilt.create_react_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_agent

            with patch("src.agents.analyst.get_llm"):
                state = _make_state()
                result = await analyst_node(state)

                assert result["agent_outputs"][0]["agent"] == "analyst"
                assert result["agent_outputs"][0]["output"] == "Key insight: AI is growing."

    @pytest.mark.asyncio
    async def test_analyst_formats_previous_data(self):
        """Analyst should format previous outputs into context."""
        from src.agents.analyst import analyst_node

        mock_result = {"messages": [MagicMock(content="Structured analysis.")]}

        with patch("langgraph.prebuilt.create_react_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_agent

            with patch("src.agents.analyst.get_llm"):
                state = _make_state(
                    agent_outputs=[
                        {"agent": "researcher", "output": "Raw research data."},
                    ],
                )
                result = await analyst_node(state)

                assert result["agent_outputs"][0]["agent"] == "analyst"

    @pytest.mark.asyncio
    async def test_analyst_handles_empty_messages(self):
        """Analyst should return 'No analysis produced.' when messages are empty."""
        from src.agents.analyst import analyst_node

        mock_result = {"messages": []}

        with patch("langgraph.prebuilt.create_react_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_agent

            with patch("src.agents.analyst.get_llm"):
                state = _make_state()
                result = await analyst_node(state)

                assert result["agent_outputs"][0]["output"] == "No analysis produced."


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
