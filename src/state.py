"""Shared state definition for the multi-agent orchestrator."""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentOutput(TypedDict):
    """Output from an individual agent."""

    agent: str
    output: str


class OrchestratorState(TypedDict):
    """Shared state across all agents in the orchestration graph.

    Attributes:
        messages: Conversation history (auto-appended via add_messages reducer).
        task: The original user task/query.
        plan: Supervisor's execution plan (list of steps).
        current_agent: Which agent is currently active.
        agent_outputs: Accumulated outputs from each agent.
        iteration: Current routing iteration (safety bound).
        final_output: The assembled final response.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    plan: list[str]
    current_agent: str
    agent_outputs: Annotated[list[AgentOutput], lambda a, b: a + b]
    iteration: int
    final_output: str
