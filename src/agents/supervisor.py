"""Supervisor agent — task decomposition, routing, and evaluation.

The Supervisor is the brain of the orchestrator. It:
1. Breaks down the user's task into a plan
2. Routes each step to the appropriate specialist agent
3. Evaluates whether the task is complete or needs more work
"""

from collections import Counter

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm, settings
from src.state import OrchestratorState

MAX_ITERATIONS = settings.max_agent_iterations
MAX_CALLS_PER_AGENT = 2

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a multi-agent team. Your job is to:
1. Follow the plan step by step
2. Route to the correct specialist for the current step
3. Move to the next step once the current one is done — do NOT repeat an agent for the same purpose
4. Respond FINISH as soon as the writer has produced the final output

Available agents:
- **researcher**: Searches the web for current information, gathers sources and data.
- **analyst**: Processes data, extracts key points, performs calculations, structures information.
- **writer**: Synthesizes information into well-written content, reports, and summaries.

Current state:
- Task: {task}
- Plan: {plan}
- Work completed so far: {completed_work}
- Agent call counts: {agent_counts}
- Iteration: {iteration}/{max_iterations}

Rules:
- Route to ONE agent at a time.
- NEVER call the same agent more than {max_calls_per_agent} times. If an agent has reached its limit, skip it and move on.
- Once the writer has produced output, respond with FINISH. Do not call the writer again to rewrite.
- If iteration is close to the limit, respond with FINISH immediately.
- Prefer moving forward over perfecting previous steps.

Respond with EXACTLY one of: researcher, analyst, writer, FINISH
Then on a new line, briefly explain why."""


async def supervisor_node(state: OrchestratorState) -> dict:
    """Supervisor node: routes to the next agent or finishes."""
    llm = get_llm(temperature=0)
    iteration = state.get("iteration", 0)

    # Count how many times each agent has been called
    agent_counts = Counter(out["agent"] for out in state.get("agent_outputs", []))

    # Hard stop: max iterations
    if iteration >= MAX_ITERATIONS:
        return {
            "current_agent": "FINISH",
            "iteration": iteration + 1,
            "messages": [
                AIMessage(content="[Supervisor] Max iterations reached. Finalizing.")
            ],
        }

    # Hard stop: if writer already produced output, we're done
    if agent_counts.get("writer", 0) >= 1 and state.get("final_output"):
        return {
            "current_agent": "FINISH",
            "iteration": iteration + 1,
            "messages": [
                AIMessage(
                    content="[Supervisor] Writer has produced output. Finalizing."
                )
            ],
        }

    # Enforce pipeline order: every agent must be called at least once
    # before moving forward: researcher -> analyst -> writer
    required_order = ["researcher", "analyst", "writer"]
    for agent in required_order:
        if agent_counts.get(agent, 0) == 0:
            return {
                "current_agent": agent,
                "iteration": iteration + 1,
                "messages": [
                    AIMessage(
                        content=f"[Supervisor] Routing to: {agent} (required pipeline step)"
                    )
                ],
            }

    # Format completed work for context
    completed_work = (
        "\n".join(
            f"- [{out['agent']}]: {out['output'][:200]}..."
            for out in state.get("agent_outputs", [])
        )
        or "None yet."
    )

    plan_str = (
        "\n".join(f"{i + 1}. {step}" for i, step in enumerate(state.get("plan", [])))
        or "No plan yet."
    )

    agent_counts_str = (
        ", ".join(
            f"{agent}: {count}/{MAX_CALLS_PER_AGENT}"
            for agent, count in agent_counts.items()
        )
        or "No agents called yet."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("human", "Decide the next step."),
        ]
    )

    response = await llm.ainvoke(
        prompt.format_messages(
            task=state["task"],
            plan=plan_str,
            completed_work=completed_work,
            agent_counts=agent_counts_str,
            iteration=iteration,
            max_iterations=MAX_ITERATIONS,
            max_calls_per_agent=MAX_CALLS_PER_AGENT,
        )
    )

    # Parse the response to extract agent name
    response_text = response.content.strip()
    first_line = response_text.split("\n")[0].strip().lower()

    next_agent = "FINISH"
    for agent_name in ["researcher", "analyst", "writer", "finish"]:
        if agent_name in first_line:
            next_agent = agent_name if agent_name != "finish" else "FINISH"
            break

    # Enforce: block agent if it already hit the call limit
    if (
        next_agent != "FINISH"
        and agent_counts.get(next_agent, 0) >= MAX_CALLS_PER_AGENT
    ):
        # If the LLM tried to call an exhausted agent, auto-advance
        # If all agents are exhausted, finish
        available = [
            a
            for a in ["researcher", "analyst", "writer"]
            if agent_counts.get(a, 0) < MAX_CALLS_PER_AGENT
        ]
        if available:
            next_agent = available[-1]  # pick the latest in the pipeline
        else:
            next_agent = "FINISH"

    return {
        "current_agent": next_agent,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(content=f"[Supervisor] Routing to: {next_agent}\n{response_text}")
        ],
    }


async def plan_task(state: OrchestratorState) -> dict:
    """Initial planning step: decompose the task into a plan.

    Called once at the beginning of the orchestration.
    """
    llm = get_llm(temperature=0)

    plan_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a task planner. Break down the following task into 2-4 clear, "
                    "sequential steps. Each step should be assignable to one of: "
                    "researcher, analyst, writer.\n\n"
                    "Respond with ONLY a numbered list of steps, nothing else."
                ),
            ),
            ("human", "{task}"),
        ]
    )

    response = await llm.ainvoke(plan_prompt.format_messages(task=state["task"]))

    # Parse numbered steps
    steps = [
        line.strip().lstrip("0123456789.").strip()
        for line in response.content.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    return {
        "plan": steps or ["Research the topic", "Analyze findings", "Write the output"],
        "iteration": 0,
        "messages": [AIMessage(content=f"[Planner] Created plan:\n{response.content}")],
    }
