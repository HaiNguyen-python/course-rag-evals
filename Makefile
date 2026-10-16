.PHONY: install test lint typecheck format index eval compare sweep up up-llm down

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

sweep: index
	course-rag sweep --charts docs/img

format:
	ruff format .
	ruff check --fix .

index:
	course-rag index

# Fails (exit 1) if the hybrid setup drops below evals/thresholds.yaml
eval: index
	course-rag eval

compare: index
	course-rag eval --compare --charts docs/img

up:
	docker compose up --build

# Local LLM: pulls the model once, then answers with it
up-llm:
	docker compose --profile llm up -d ollama
	docker compose exec ollama ollama pull llama3.1:8b
	GENERATOR=openai docker compose --profile llm up --build

down:
	docker compose --profile llm down -v
