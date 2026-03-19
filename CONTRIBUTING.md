# Contributing

Thanks for your interest in contributing to Multi-Agent Orchestrator!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/multi-agent-orchestrator.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys

## Development

### Running Tests

```bash
# Unit tests (no API keys needed)
pytest tests/ -v -m "not integration"

# All tests (requires API keys)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term
```

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Adding a New Agent

1. Create `src/agents/your_agent.py` — implement an async node function that takes `OrchestratorState` and returns a dict
2. Add any new tools in `src/tools/`
3. Register the agent node in `src/orchestrator.py`
4. Update the Supervisor prompt in `src/agents/supervisor.py` to know about the new agent
5. Add tests in `tests/test_agents.py`
6. Update `README.md`

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add code reviewer agent with AST analysis
fix: prevent infinite loop in supervisor routing
docs: update architecture diagram with new agent
test: add integration tests for full pipeline
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes
3. Ensure tests pass: `pytest tests/ -v`
4. Ensure lint passes: `ruff check src/ tests/`
5. Push and open a PR against `main`

## Questions?

Open an issue or reach out to [@reiquileut](https://github.com/reiquileut).
