.PHONY: install dev test lint format fmt-all clean publish

install:
	uv pip install -e .

dev:
	uv sync --all-extras

test:
	uv run pytest -v

lint:
	ruff check .
	ty check .

format:
	ruff format .
	ruff check . --fix

fmt-all:
	deno fmt --group dev .

clean:
	rm -rf build/ dist/ *.egg-info treegrad.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

publish: clean
	uv build
	twine upload dist/*
