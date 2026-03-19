"""CLI entry point for the Multi-Agent Orchestrator.

Usage:
    python -m src.cli "Your task description here"
    python -m src.cli --task "Research AI frameworks" --verbose
"""

import argparse
import asyncio
import uuid

from src.orchestrator import orchestrator


async def run_task(task: str, verbose: bool = False) -> str:
    """Execute a task through the multi-agent orchestrator.

    Args:
        task: The task description to execute.
        verbose: If True, print intermediate agent outputs.

    Returns:
        The final output string.
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

    # Stream events for verbose output
    if verbose:
        async for event in orchestrator.astream_events(
            {"task": task, "messages": []},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chain_end" and event.get("name") in (
                "planner", "supervisor", "researcher", "analyst", "writer"
            ):
                print(f"\n--- [{event['name'].upper()}] ---")
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    agent_out = output.get("agent_outputs", [])
                    for o in agent_out:
                        print(f"  {o['output'][:300]}")
                    if output.get("current_agent"):
                        print(f"  → Next: {output['current_agent']}")
    else:
        result = await orchestrator.ainvoke(
            {"task": task, "messages": []},
            config=config,
        )

    # Fetch final state
    final_state = await orchestrator.aget_state(config)
    final_output = final_state.values.get("final_output", "No output generated.")

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL OUTPUT:")
        print(f"{'='*60}")
        print(final_output)

    return final_output


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task description to execute",
    )
    parser.add_argument(
        "--task", "-t",
        dest="task_flag",
        help="Task description (alternative to positional arg)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show intermediate agent outputs",
    )

    args = parser.parse_args()
    task = args.task or args.task_flag

    if not task:
        parser.error("Please provide a task description.")

    asyncio.run(run_task(task, verbose=args.verbose))


if __name__ == "__main__":
    main()
