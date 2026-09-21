.PHONY: dev test lint

dev:
	uv run python -m src.main

test:
	uv run pytest
	uv run ruff check src tests
	uv run mypy src

lint:
	uv run ruff check src tests
	uv run mypy src
