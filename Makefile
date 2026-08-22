.PHONY: publish test lint format

publish:
	uv build
	uv publish

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
