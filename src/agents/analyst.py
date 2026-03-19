"""Analyst agent — data processing, extraction, and structured analysis.

The Analyst agent is responsible for:
- Processing raw research data into structured formats
- Extracting key metrics, facts, and insights
- Performing calculations when needed
- Fact-checking and validating information
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_llm
from src.state import OrchestratorState
from src.tools.calculator import calculate
from src.tools.text_processing import summarize_text, extract_key_points

ANALYST_SYSTEM_PROMPT = """You are a Data Analyst Specialist. Your role is to process and \
structure information into clear, actionable insights.

Task: {task}
Available data from previous agents:
{previous_data}

Instructions:
1. Analyze the data provided by previous agents
2. Extract key metrics, facts, and patterns
3. Use the calculator for any numerical analysis
4. Use summarize_text or extract_key_points for text processing
5. Structure your output with clear headers and bullet points

Output a structured analysis that the Writer can use to produce the final deliverable."""


async def analyst_node(state: OrchestratorState) -> dict:
    """Analyst node: processes data and produces structured analysis.

    Uses tools for calculation and text processing.
    """
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=0)

    previous_data = (
        "\n\n".join(
            f"=== From {o['agent']} ===\n{o['output']}"
            for o in state.get("agent_outputs", [])
        )
        or "No previous data available."
    )

    analyst_tools = [calculate, summarize_text, extract_key_points]

    react_agent = create_react_agent(
        llm,
        analyst_tools,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    ANALYST_SYSTEM_PROMPT.format(
                        task=state["task"],
                        previous_data=previous_data,
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        ),
    )

    result = await react_agent.ainvoke(
        {"messages": [("human", f"Analyze the data for: {state['task']}")]}
    )

    output = (
        result["messages"][-1].content
        if result["messages"]
        else "No analysis produced."
    )

    return {
        "agent_outputs": [{"agent": "analyst", "output": output}],
        "messages": [AIMessage(content=f"[Analyst] {output}")],
    }
