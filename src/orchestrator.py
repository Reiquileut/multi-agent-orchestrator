"""Orchestrator — LangGraph graph builder and compiler.

This module builds the StateGraph that connects the Supervisor to
the specialist agents via conditional edges. The Supervisor pattern
ensures centralized task routing with iterative refinement.

Graph structure:
    START → planner → supervisor ←→ [researcher | analyst | writer] → assemble → END
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from src.state import OrchestratorState
from src.agents.supervisor import supervisor_node, plan_task
from src.agents.researcher import researcher_node
from src.agents.analyst import analyst_node
from src.agents.writer import writer_node


def _route_from_supervisor(
    state: OrchestratorState,
) -> Literal["researcher", "analyst", "writer", "assemble"]:
    """Route based on Supervisor's decision.

    Returns the next node name in the graph.
    """
    next_agent = state.get("current_agent", "FINISH")

    if next_agent == "FINISH":
        return "assemble"
    elif next_agent in ("researcher", "analyst", "writer"):
        return next_agent
    else:
        return "assemble"


async def assemble_output(state: OrchestratorState) -> dict:
    """Final assembly node: extracts the final output or generates a fallback.

    If the Writer has produced a final_output, use it.
    Otherwise, compile a summary from all agent outputs.
    """
    if state.get("final_output"):
        return {"final_output": state["final_output"]}

    # Fallback: compile from agent outputs
    compiled = "\n\n".join(
        f"## {o['agent'].title()}\n{o['output']}"
        for o in state.get("agent_outputs", [])
    )

    return {
        "final_output": compiled or "No output was generated.",
        "messages": [AIMessage(content="[Assembler] Final output compiled.")],
    }


def build_graph(with_memory: bool = True) -> StateGraph:
    """Build and compile the multi-agent orchestrator graph.

    Args:
        with_memory: If True, uses MemorySaver for checkpointing (enables
                     conversation persistence across invocations).

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    builder = StateGraph(OrchestratorState)

    # --- Register nodes ---
    builder.add_node("planner", plan_task)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)
    builder.add_node("assemble", assemble_output)

    # --- Define edges ---
    # Entry: plan the task first
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")

    # Supervisor routes conditionally
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "assemble": "assemble",
        },
    )

    # Each specialist returns to the Supervisor for re-evaluation
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("writer", "supervisor")

    # Assembly is the terminal node
    builder.add_edge("assemble", END)

    # --- Compile ---
    checkpointer = MemorySaver() if with_memory else None
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# Pre-built graph instance for convenience
orchestrator = build_graph(with_memory=True)
