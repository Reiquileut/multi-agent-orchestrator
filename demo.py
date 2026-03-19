"""Demo script — runs the full orchestration pipeline with mock LLM responses.

No API keys required. Shows the Supervisor routing pattern, agent handoffs,
and the complete execution flow for demonstration purposes.

Usage:
    python demo.py
"""

import asyncio
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.state import OrchestratorState
from src.orchestrator import build_graph

# ── Mock Responses ───────────────────────────────────────────────────────────
# Simulate realistic LLM responses for each agent phase

MOCK_PLAN = """1. Research the current state of AI agent frameworks and their adoption
2. Analyze the key features, trade-offs, and market positioning of each framework
3. Write a structured comparison report with recommendations"""

MOCK_SUPERVISOR_RESPONSES = [
    "researcher\nNeed to gather current data on AI agent frameworks before analysis.",
    "analyst\nResearch data collected. Now need structured analysis of the findings.",
    "writer\nAnalysis complete. Ready to synthesize into final report.",
    "FINISH\nThe writer has produced a comprehensive comparison report.",
]

MOCK_RESEARCH = """## Research Findings: AI Agent Frameworks (2025)

**1. LangGraph (LangChain)**
- Most popular framework for stateful agent orchestration
- Supervisor and hierarchical patterns built-in
- Strong ecosystem: LangSmith tracing, LangServe deployment
- Used by: Elastic, Replit, Rakuten

**2. CrewAI**
- Role-based multi-agent framework
- Simpler API, faster prototyping
- Built-in delegation between agents
- Growing community, 45K+ GitHub stars

**3. AutoGen (Microsoft)**
- Conversational multi-agent framework
- Strong code generation capabilities
- Enterprise backing from Microsoft
- Complex setup, steeper learning curve

Sources: GitHub stars, framework docs, industry surveys (2025)."""

MOCK_ANALYSIS = """## Structured Analysis

| Criteria        | LangGraph     | CrewAI        | AutoGen       |
|----------------|---------------|---------------|---------------|
| Flexibility    | ★★★★★        | ★★★☆☆        | ★★★★☆        |
| Ease of Use    | ★★★☆☆        | ★★★★★        | ★★☆☆☆        |
| Production     | ★★★★★        | ★★★☆☆        | ★★★★☆        |
| Community      | ★★★★★        | ★★★★☆        | ★★★☆☆        |
| Documentation  | ★★★★☆        | ★★★★☆        | ★★★☆☆        |

**Key Insights:**
- LangGraph dominates production use cases due to state management and checkpointing
- CrewAI wins for rapid prototyping and simpler multi-agent setups
- AutoGen excels at code-heavy tasks but has higher complexity overhead
- Market trend: convergence toward stateful, graph-based orchestration"""

MOCK_REPORT = """# AI Agent Frameworks Comparison Report (2025)

## Executive Summary
The AI agent framework landscape in 2025 is dominated by three major players:
LangGraph, CrewAI, and AutoGen. Each serves distinct use cases, from production
enterprise systems to rapid prototyping.

## Framework Overview

### LangGraph (by LangChain)
LangGraph has emerged as the industry standard for production-grade agent systems.
Its StateGraph architecture provides explicit state management, durable execution,
and built-in support for human-in-the-loop workflows. The Supervisor pattern enables
centralized routing between specialized agents with conditional edges.

**Best for:** Production systems, complex multi-step workflows, enterprise deployments.

### CrewAI
CrewAI takes a role-based approach where agents are defined by their role, goal,
and backstory. This makes it intuitive to set up collaborative agent teams. The
framework handles delegation automatically and provides a simpler mental model
than graph-based alternatives.

**Best for:** Rapid prototyping, content generation pipelines, team-based agent workflows.

### AutoGen (by Microsoft)
AutoGen focuses on conversational patterns between agents, with strong code generation
and execution capabilities. Its enterprise backing from Microsoft provides stability
but the setup complexity is notably higher than alternatives.

**Best for:** Code-heavy automation, research workflows, Microsoft ecosystem integration.

## Recommendation
- **For production:** LangGraph — unmatched control, observability, and reliability
- **For prototypes:** CrewAI — fastest time-to-demo with intuitive API
- **For code tasks:** AutoGen — strongest code generation and execution loop

## Key Takeaways
1. The market is converging toward stateful, graph-based orchestration
2. Supervisor patterns are becoming the standard for multi-agent routing
3. Observability (LangSmith, tracing) is now a must-have, not a nice-to-have
4. All three frameworks support tool-calling with structured Pydantic schemas"""

# ── Mock LLM Factory ────────────────────────────────────────────────────────

_supervisor_call_count = 0


def _mock_llm_factory(temperature=0):
    """Create a mock LLM that returns predetermined responses."""
    global _supervisor_call_count

    mock = MagicMock()

    async def mock_ainvoke(messages, **kwargs):
        global _supervisor_call_count
        response = MagicMock()

        # Detect which agent is calling based on prompt content
        prompt_text = str(messages).lower()

        if "task planner" in prompt_text:
            response.content = MOCK_PLAN
        elif "content writer specialist" in prompt_text:
            response.content = MOCK_REPORT
        elif "research specialist" in prompt_text:
            response.content = MOCK_RESEARCH
        elif "data analyst specialist" in prompt_text:
            response.content = MOCK_ANALYSIS
        elif "supervisor" in prompt_text or "which agent" in prompt_text:
            idx = min(_supervisor_call_count, len(MOCK_SUPERVISOR_RESPONSES) - 1)
            response.content = MOCK_SUPERVISOR_RESPONSES[idx]
            _supervisor_call_count += 1
        else:
            response.content = "FINISH\nTask complete."

        return response

    mock.ainvoke = mock_ainvoke
    return mock


# ── Demo Runner ─────────────────────────────────────────────────────────────

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "purple": "\033[35m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "red": "\033[31m",
}


def c(text: str, color: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


async def run_demo():
    global _supervisor_call_count
    _supervisor_call_count = 0

    task = "Compare the top 3 AI agent frameworks in 2025 and write a report"

    print()
    print(c("=" * 70, "dim"))
    print(c("  🤖 Multi-Agent Orchestrator — Demo (Mock Mode)", "bold"))
    print(c("=" * 70, "dim"))
    print()
    print(f"  {c('Task:', 'bold')} {task}")
    print()
    print(c("  No API keys required — using simulated agent responses.", "dim"))
    print(c("-" * 70, "dim"))

    # Patch get_llm to return our mock
    with patch("src.agents.supervisor.get_llm", side_effect=_mock_llm_factory):
        with patch("src.agents.researcher.get_llm", side_effect=_mock_llm_factory):
            with patch("src.agents.analyst.get_llm", side_effect=_mock_llm_factory):
                with patch("src.agents.writer.get_llm", side_effect=_mock_llm_factory):
                    # Also patch create_react_agent for researcher/analyst
                    async def mock_react_invoke(input_dict):
                        messages = input_dict.get("messages", [])
                        task_text = str(messages)

                        if "research" in task_text.lower():
                            content = MOCK_RESEARCH
                        elif "analy" in task_text.lower():
                            content = MOCK_ANALYSIS
                        else:
                            content = "Mock agent output."

                        return {"messages": [AIMessage(content=content)]}

                    mock_react = MagicMock()
                    mock_react.ainvoke = mock_react_invoke

                    with patch("src.agents.researcher.create_react_agent", return_value=mock_react):
                        with patch("langgraph.prebuilt.create_react_agent", return_value=mock_react):
                            graph = build_graph(with_memory=False)

                            config = {"configurable": {"thread_id": "demo-001"}}

                            result = await graph.ainvoke(
                                {"task": task, "messages": []},
                                config=config,
                            )

    # ── Display Results ──────────────────────────────────────────────────
    print()
    print(c("  📋 EXECUTION TRACE", "bold"))
    print(c("-" * 70, "dim"))

    agent_icons = {
        "researcher": ("🔍", "cyan"),
        "analyst": ("📊", "yellow"),
        "writer": ("✍️", "green"),
    }

    for i, output in enumerate(result.get("agent_outputs", []), 1):
        agent = output["agent"]
        icon, color = agent_icons.get(agent, ("🤖", "purple"))
        print(f"\n  {c(f'Step {i}:', 'bold')} {icon} {c(agent.upper(), color)}")
        # Show first 3 lines of output
        lines = output["output"].strip().split("\n")[:3]
        for line in lines:
            print(f"    {c(line, 'dim')}")
        if len(output["output"].strip().split("\n")) > 3:
            print(f"    {c('...', 'dim')}")

    # Final output
    print()
    print(c("=" * 70, "dim"))
    print(c("  📄 FINAL OUTPUT", "bold"))
    print(c("=" * 70, "dim"))
    print()
    print(result.get("final_output", "No output."))
    print()
    print(c("-" * 70, "dim"))
    print(c(f"  ✅ Completed with {len(result.get('agent_outputs', []))} agent steps", "green"))
    print(c(f"  📊 Graph: Planner → Supervisor → Researcher → Analyst → Writer → Assemble", "dim"))
    print(c("-" * 70, "dim"))
    print()


if __name__ == "__main__":
    asyncio.run(run_demo())
