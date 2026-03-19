"""Researcher agent — web search and information gathering.

The Researcher agent is responsible for:
- Searching the web for relevant, current information
- Scraping specific URLs for detailed content
- Gathering and organizing source material for other agents
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from src.config import get_llm
from src.state import OrchestratorState
from src.tools.search import web_search, scrape_url

RESEARCHER_SYSTEM_PROMPT = """You are a Research Specialist. Your role is to gather accurate, \
current information from the web.

Task: {task}
Plan: {plan}
Previous findings: {previous_findings}

Instructions:
1. Use web_search to find relevant information
2. Use scrape_url if you need deeper content from a specific source
3. Focus on facts, data, and credible sources
4. Synthesize your findings into a clear research brief

Be thorough but concise. Cite your sources."""

# Build a ReAct agent with search tools
_researcher_tools = [web_search, scrape_url]


async def researcher_node(state: OrchestratorState) -> dict:
    """Researcher node: gathers information via web search.

    Uses a ReAct agent internally for tool-calling autonomy.
    """
    llm = get_llm(temperature=0)

    previous = (
        "\n".join(
            f"- [{o['agent']}]: {o['output'][:150]}"
            for o in state.get("agent_outputs", [])
        )
        or "None."
    )

    plan_str = ", ".join(state.get("plan", []))

    # Create a standalone ReAct agent for the researcher
    react_agent = create_react_agent(
        llm,
        _researcher_tools,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    RESEARCHER_SYSTEM_PROMPT.format(
                        task=state["task"],
                        plan=plan_str,
                        previous_findings=previous,
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        ),
    )

    result = await react_agent.ainvoke(
        {"messages": [("human", f"Research this task: {state['task']}")]}
    )

    # Extract the final response
    output = result["messages"][-1].content if result["messages"] else "No findings."

    return {
        "agent_outputs": [{"agent": "researcher", "output": output}],
        "messages": [AIMessage(content=f"[Researcher] {output}")],
    }
