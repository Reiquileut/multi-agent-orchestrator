.PHONY: install run demo test lint clean docker

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	streamlit run app.py

demo:
	python demo.py

cli:
	python -m src.cli "$(TASK)" --verbose

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

docker:
	docker compose up --build

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
