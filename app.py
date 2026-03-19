"""Streamlit application for the Multi-Agent Orchestrator.

Run with: streamlit run app.py
"""

import asyncio
import uuid

import streamlit as st
from src.config import settings

# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Orchestrator",
    page_icon="https://img.icons8.com/fluency-systems-regular/48/workflow.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Sidebar nav items */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        color: #b0b0b0;
        font-size: 0.9rem;
        text-decoration: none;
        transition: background 0.15s;
        margin-bottom: 2px;
    }
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #fff;
    }
    .nav-item.active {
        background: rgba(108, 99, 255, 0.12);
        color: #fff;
    }
    .nav-icon {
        width: 18px;
        height: 18px;
        opacity: 0.7;
    }
    .nav-section {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #666;
        padding: 1rem 0.75rem 0.35rem;
    }

    /* Logo area */
    .sidebar-logo {
        font-size: 1.1rem;
        font-weight: 700;
        color: #fff;
        padding: 0.25rem 0.75rem 1.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 0.75rem;
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Greeting */
    .greeting-sub {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.15rem;
    }
    .greeting-main {
        font-size: 1.75rem;
        font-weight: 700;
        color: #fafafa;
        margin-bottom: 1.5rem;
    }

    /* Task cards */
    .task-card {
        background: #1a1d23;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.25rem;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.2s, background 0.2s;
        cursor: pointer;
    }
    .task-card:hover {
        border-color: rgba(108, 99, 255, 0.4);
        background: #1e2128;
    }
    .task-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.35rem;
    }
    .task-card-desc {
        font-size: 0.75rem;
        color: #888;
        line-height: 1.4;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-running {
        background: rgba(108, 99, 255, 0.15);
        color: #9d97ff;
    }
    .status-done {
        background: rgba(45, 183, 105, 0.15);
        color: #2db769;
    }
    .status-error {
        background: rgba(255, 75, 75, 0.15);
        color: #ff6b6b;
    }

    /* Agent step */
    .agent-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        font-size: 0.85rem;
    }
    .agent-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 1.5rem 0 0.75rem;
    }

    /* Config panel in sidebar */
    .config-label {
        font-size: 0.75rem;
        color: #888;
        font-weight: 500;
        margin-bottom: 2px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: #6C63FF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button[kind="primary"]:hover {
        background: #5a52e0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────────────────────────


def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []
    if "running" not in st.session_state:
        st.session_state.running = False
    if "page" not in st.session_state:
        st.session_state.page = "orchestrator"


init_session()

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">Multi-Agent Orchestrator</div>', unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="nav-section">Workspace</div>', unsafe_allow_html=True)

    if st.button("Orchestrator", use_container_width=True, key="nav_orch"):
        st.session_state.page = "orchestrator"
    if st.button("History", use_container_width=True, key="nav_hist"):
        st.session_state.page = "history"
    if st.button("Architecture", use_container_width=True, key="nav_arch"):
        st.session_state.page = "architecture"

    st.markdown('<div class="nav-section">Configuration</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "Provider",
        ["openai", "anthropic"],
        index=0,
        label_visibility="collapsed",
        help="LLM provider",
    )

    # Show current model
    default_model = "gpt-5.4-nano" if provider == "openai" else "claude-sonnet-4-5-20250514"
    model = st.text_input("Model", value=default_model, label_visibility="collapsed")

    # Status indicator
    has_llm_key = bool(
        (provider == "openai" and settings.openai_api_key and settings.openai_api_key != "your-openai-key-here")
        or (provider == "anthropic" and settings.anthropic_api_key and settings.anthropic_api_key != "your-anthropic-key-here")
    )
    has_tavily = bool(settings.tavily_api_key and settings.tavily_api_key != "your-tavily-key-here")

    st.markdown('<div class="nav-section">Status</div>', unsafe_allow_html=True)

    if has_llm_key and has_tavily:
        st.markdown("API Keys: **Connected**", help="Keys loaded from .env file")
    else:
        missing = []
        if not has_llm_key:
            missing.append(f"{provider.title()} API key")
        if not has_tavily:
            missing.append("Tavily API key")
        st.markdown(f"Missing: {', '.join(missing)}")
        st.caption("Configure in .env file")

    st.markdown("---")
    if st.button("New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

# ── Orchestrator Execution ──────────────────────────────────────────────────


async def run_orchestrator(task: str, provider: str, model: str):
    """Run the orchestrator using keys from .env."""
    import os

    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model

    from importlib import reload
    import src.config
    reload(src.config)

    from src.orchestrator import build_graph

    graph = build_graph(with_memory=False)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    result = await graph.ainvoke(
        {"task": task, "messages": []},
        config=config,
    )

    return result


# ── Pages ───────────────────────────────────────────────────────────────────

if st.session_state.page == "orchestrator":

    st.markdown('<p class="greeting-sub">Workspace</p>', unsafe_allow_html=True)
    st.markdown('<p class="greeting-main">Orchestrator</p>', unsafe_allow_html=True)

    # Example task cards
    st.markdown('<p class="section-header">Quick start</p>', unsafe_allow_html=True)

    example_tasks = {
        "Research Report": {
            "desc": "Compare AI agent frameworks with pros, cons, and recommendations",
            "task": "Research the top 3 AI agent frameworks in 2025 and write a comparison report with pros and cons",
        },
        "Market Analysis": {
            "desc": "Analyze job market trends and identify the most in-demand skills",
            "task": "Analyze current trends in the AI job market and summarize the most in-demand skills",
        },
        "Technical Blog": {
            "desc": "Write an in-depth blog post about multi-agent AI systems",
            "task": "Write a technical blog post explaining how multi-agent AI systems work, with real-world examples",
        },
    }

    cols = st.columns(3)
    selected_example = None
    for col, (title, data) in zip(cols, example_tasks.items()):
        with col:
            if st.button(
                f"**{title}**\n\n{data['desc']}",
                use_container_width=True,
                key=f"example_{title}",
            ):
                selected_example = data["task"]

    # Task input
    st.markdown('<p class="section-header">Describe your task</p>', unsafe_allow_html=True)

    task_input = st.text_area(
        "Task",
        value=selected_example or "",
        height=100,
        placeholder="e.g., Research the latest developments in quantum computing and write a summary report...",
        label_visibility="collapsed",
    )

    # Run button
    can_run = has_llm_key and has_tavily
    run_col, info_col = st.columns([1, 3])

    with run_col:
        run_clicked = st.button(
            "Run",
            type="primary",
            disabled=st.session_state.running or not can_run,
            use_container_width=True,
        )

    if not can_run:
        with info_col:
            st.caption("Configure API keys in .env to run the orchestrator.")

    if run_clicked:
        if not task_input.strip():
            st.warning("Please enter a task description.")
        else:
            st.session_state.running = True

            status_container = st.container()

            with status_container:
                with st.status("Running orchestrator...", expanded=True) as status:
                    st.write("Planning task decomposition...")

                    try:
                        result = asyncio.run(
                            run_orchestrator(task_input, provider, model)
                        )

                        agent_colors = {
                            "researcher": "#4A9EFF",
                            "analyst": "#FFB84D",
                            "writer": "#2DB769",
                        }

                        for output in result.get("agent_outputs", []):
                            agent = output["agent"]
                            color = agent_colors.get(agent, "#888")
                            st.markdown(
                                f'<div class="agent-step">'
                                f'<span class="agent-dot" style="background:{color}"></span>'
                                f'<span><strong>{agent.title()}</strong> completed</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        status.update(
                            label="Task completed",
                            state="complete",
                        )

                    except Exception as e:
                        status.update(label="Error", state="error")
                        st.error(f"Error: {e}")
                        st.session_state.running = False
                        st.stop()

            # Final output
            st.markdown('<p class="section-header">Result</p>', unsafe_allow_html=True)

            final = result.get("final_output", "No output generated.")
            st.markdown(final)

            # Agent details
            st.markdown('<p class="section-header">Agent Details</p>', unsafe_allow_html=True)

            for output in result.get("agent_outputs", []):
                agent = output["agent"]
                with st.expander(f"{agent.title()} Output"):
                    st.markdown(output["output"])

            # Save to history
            st.session_state.history.append({
                "task": task_input,
                "output": final,
                "agents": [o["agent"] for o in result.get("agent_outputs", [])],
            })

            st.session_state.running = False


elif st.session_state.page == "history":

    st.markdown('<p class="greeting-sub">Workspace</p>', unsafe_allow_html=True)
    st.markdown('<p class="greeting-main">History</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No tasks have been executed yet. Run a task from the Orchestrator page.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - i
            with st.expander(f"Task {idx}: {entry['task'][:80]}"):
                st.markdown(f"**Agents used:** {', '.join(entry['agents'])}")
                st.markdown("---")
                st.markdown(entry["output"])


elif st.session_state.page == "architecture":

    st.markdown('<p class="greeting-sub">Workspace</p>', unsafe_allow_html=True)
    st.markdown('<p class="greeting-main">Architecture</p>', unsafe_allow_html=True)

    st.markdown("""
The orchestrator uses a **Supervisor pattern** built with LangGraph. The Supervisor
coordinates specialist agents through a shared state, routing tasks dynamically
based on what each step requires.
""")

    st.markdown('<p class="section-header">Execution Flow</p>', unsafe_allow_html=True)

    flow_data = [
        ("1", "Planner", "Decomposes the user task into 2-4 sequential steps", "#6C63FF"),
        ("2", "Supervisor", "Evaluates progress and routes to the next specialist", "#6C63FF"),
        ("3", "Researcher", "Gathers information via web search (Tavily API)", "#4A9EFF"),
        ("4", "Analyst", "Processes data, extracts metrics, structures analysis", "#FFB84D"),
        ("5", "Writer", "Synthesizes all material into the final deliverable", "#2DB769"),
        ("6", "Assemble", "Compiles the final output and returns to user", "#6C63FF"),
    ]

    for step, name, desc, color in flow_data:
        st.markdown(
            f'<div class="agent-step">'
            f'<span class="agent-dot" style="background:{color}"></span>'
            f'<strong>{step}. {name}</strong> &mdash; {desc}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-header">Agent Tools</p>', unsafe_allow_html=True)

    tools_data = {
        "Researcher": ["web_search -- Tavily web search", "scrape_url -- Extract content from URLs"],
        "Analyst": ["calculate -- Safe math evaluation", "summarize_text -- Extractive summarization", "extract_key_points -- Key point extraction"],
        "Writer": ["No tools -- Pure LLM generation with temperature 0.3"],
    }

    for agent, tools in tools_data.items():
        with st.expander(agent):
            for t in tools:
                st.markdown(f"- `{t}`")

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Built with [LangGraph](https://github.com/langchain-ai/langgraph) · "
    "[Streamlit](https://streamlit.io/) · "
    "[Source](https://github.com/reiquileut/multi-agent-orchestrator)"
)
