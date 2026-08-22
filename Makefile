.PHONY: publish test

publish:
	uv build
	uv publish

test:
	uv run pytest
