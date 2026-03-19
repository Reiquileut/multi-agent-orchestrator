"""Integration tests for the orchestrator graph structure."""

import pytest
from src.orchestrator import build_graph, _route_from_supervisor


class TestGraphStructure:
    """Tests for graph compilation and structure."""

    def test_graph_compiles(self):
        """Graph should compile without errors."""
        graph = build_graph(with_memory=False)
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """Graph should contain all expected nodes."""
        graph = build_graph(with_memory=False)
        node_names = set(graph.get_graph().nodes.keys())

        expected = {
            "planner",
            "supervisor",
            "researcher",
            "analyst",
            "writer",
            "assemble",
        }
        # LangGraph adds __start__ and __end__ nodes
        assert expected.issubset(node_names)


class TestRouting:
    """Tests for the routing function."""

    def test_routes_to_researcher(self):
        state = {"current_agent": "researcher"}
        assert _route_from_supervisor(state) == "researcher"

    def test_routes_to_analyst(self):
        state = {"current_agent": "analyst"}
        assert _route_from_supervisor(state) == "analyst"

    def test_routes_to_writer(self):
        state = {"current_agent": "writer"}
        assert _route_from_supervisor(state) == "writer"

    def test_routes_to_assemble_on_finish(self):
        state = {"current_agent": "FINISH"}
        assert _route_from_supervisor(state) == "assemble"

    def test_routes_to_assemble_on_unknown(self):
        state = {"current_agent": "unknown_agent"}
        assert _route_from_supervisor(state) == "assemble"

    def test_routes_to_assemble_on_empty(self):
        state = {}
        assert _route_from_supervisor(state) == "assemble"


class TestAssembleOutput:
    """Tests for the final assembly node."""

    @pytest.mark.asyncio
    async def test_returns_existing_final_output(self):
        from src.orchestrator import assemble_output

        state = {"final_output": "Already done.", "agent_outputs": []}
        result = await assemble_output(state)
        assert result["final_output"] == "Already done."

    @pytest.mark.asyncio
    async def test_compiles_from_agent_outputs(self):
        from src.orchestrator import assemble_output

        state = {
            "final_output": "",
            "agent_outputs": [
                {"agent": "researcher", "output": "Found data."},
                {"agent": "analyst", "output": "Analyzed data."},
            ],
        }
        result = await assemble_output(state)
        assert "Researcher" in result["final_output"]
        assert "Analyst" in result["final_output"]

    @pytest.mark.asyncio
    async def test_fallback_on_empty(self):
        from src.orchestrator import assemble_output

        state = {"final_output": "", "agent_outputs": []}
        result = await assemble_output(state)
        assert result["final_output"] == "No output was generated."
