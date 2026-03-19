"""Writer agent — content synthesis, formatting, and final output.

The Writer agent is responsible for:
- Synthesizing research and analysis into polished content
- Formatting output appropriate to the task (report, blog, summary, etc.)
- Ensuring clarity, coherence, and proper structure
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_llm
from src.state import OrchestratorState

WRITER_SYSTEM_PROMPT = """You are a Content Writer Specialist. Your role is to synthesize \
information into clear, well-structured content.

Original task: {task}

Data and analysis from the team:
{source_material}

Instructions:
1. Synthesize ALL provided material into a cohesive deliverable
2. Match the format to what the task requires (report, summary, blog post, etc.)
3. Use clear headers, logical flow, and professional tone
4. Include specific data points, quotes, and sources where available
5. End with key takeaways or conclusions

Produce the FINAL deliverable — this is what the user will see."""


async def writer_node(state: OrchestratorState) -> dict:
    """Writer node: produces the final polished output.

    Does not use tools — relies purely on LLM generation
    with source material from other agents.
    """
    llm = get_llm(temperature=0.3)  # Slightly creative

    source_material = "\n\n".join(
        f"=== {o['agent'].upper()} OUTPUT ===\n{o['output']}"
        for o in state.get("agent_outputs", [])
    ) or "No source material available."

    prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_SYSTEM_PROMPT.format(
            task=state["task"],
            source_material=source_material,
        )),
        ("human", "Write the final deliverable for the task."),
    ])

    response = await llm.ainvoke(prompt.format_messages())

    output = response.content

    return {
        "agent_outputs": [{"agent": "writer", "output": output}],
        "final_output": output,
        "messages": [AIMessage(content=f"[Writer] {output}")],
    }
