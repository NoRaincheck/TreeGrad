# TreeGrad — Agent Guidelines

## Project Overview

TreeGrad converts LightGBM gradient-boosted trees into differentiable decision forests that can be trained with auto-differentiation. The project uses `uv` for all Python tooling and dependency management.

## Quick Start (uv)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment & install deps
uv venv && source .venv/bin/activate        # Linux/macOS
uv venv && .venv\Scripts\activate           # Windows

# Install package + dev dependencies
uv sync --all-extras

# Or just core + torch:
uv pip install -e ".[torch]"
```

## Code Formatting & Linting

### Ruff (Python)

Run the linter and auto-fix:

```bash
ruff check .
ruff check . --fix          # apply safe fixes
ruff format .               # format Python files
```

Ruff configuration is in `pyproject.toml`:
- Line length: **120**
- Enabled rules: `E4`, `E7`, `E9`, `F`, `I001` (imports, whitespace, syntax)

### Ty (Python Type Checking)

```bash
ty check .                  # type-check the project
```

### Deno Format (Markdown & config files)

For non-Python files (markdown, JSON configs), use deno fmt:

```bash
deno fmt --group dev .      # format all supported files
```

Deno formatting is configured via `.vscode/settings.json` or a `deno.json`:
- Formatter line width: **120**
- Files included in `--fmt`: `**/*.md`

## Testing (pytest)

Tests live in `treegrad/tests/`. Run with pytest:

```bash
uv run pytest               # run all tests
uv run pytest -v            # verbose output
uv run pytest --cov=treegrad  # with coverage report
```

The test suite covers binary classification, multiclass classification, and regression.

## Dependency Groups (pyproject.toml)

| Group     | Contents                                      |
|-----------|-----------------------------------------------|
| `dev`     | ruff, ty, pytest, pytest-cov                  |
| `torch`   | torch, torchvision                            |
| `notebooks`| jupyter, matplotlib, seaborn                 |

Install groups:

```bash
uv sync --group dev       # dev tools only
uv pip install -e ".[dev]"  # editable + dev extras
uv run --all-extras pytest  # everything
```

## Running Notebooks

```bash
uv run jupyter notebook notebooks/
```

## Versioning

This project uses semantic versioning. The current major version is **2**. Bump the `version` field in `pyproject.toml`.
