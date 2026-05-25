# TreeGrad 2.0

[![PyPI version](https://badge.fury.io/py/treegrad.png)](https://badge.fury.io/py/treegrad)

`TreeGrad` implements a naive approach to converting a Gradient Boosted Tree Model to an Online trainable model. It does this by creating differentiable tree models which can be learned via auto-differentiable frameworks. `TreeGrad` is in essence an implementation of Kontschieder, Peter, et al. "Deep neural decision forests." with extensions.

## Install (uv)

```bash
# Create virtual environment & install package
uv venv && source .venv/bin/activate        # Linux/macOS
uv pip install -e ".[dev]"                   # editable + dev tools
uv pip install -e ".[torch]"                 # with PyTorch support
```

or alternatively from pypi:

```bash
pip install treegrad
```

## Run Tests (pytest)

```bash
uv run pytest -v
uv run pytest --cov=treegrad    # with coverage
```

## Lint & Format

```bash
ruff check . && ruff format .   # Python linting/formatting
deno fmt --group dev .          # Markdown/config formatting
ty check .                      # type checking
```

See [AGENTS.md](AGENTS.md) for full agent guidelines.

## Usage

```py
from sklearn.datasets import make_classification
import treegrad as tgd

X, y = make_classification(1000, n_classes=3, random_state=42)

mod = tgd.TGDClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)
mod.fit(X, y)
mod.partial_fit(X, y)  # online / incremental learning
```

## Requirements

Core dependencies:

* lightgbm
* scikit-learn
* autograd

Optional extras (via pyproject.toml):

| Group     | Contents                          |
|-----------|-----------------------------------|
| `dev`     | ruff, ty, pytest, pytest-cov      |
| `torch`   | torch, torchvision                |
| `notebooks`| jupyter, matplotlib, seaborn     |

## Results

When decision splits are reset and subsequently re-learned, TreeGrad can be competitive in performance with popular implementations (albeit an order of magnitude slower). Below is a table showing accuracy on test dataset on UCI benchmark datasets for Boosted Ensemble models (100 trees):

| Dataset  | TreeGrad  | LightGBM  | Scikit-Learn (Gradient Boosting Classifier) |
| ---------| --------- | --------- | ------------------------------------------- |
| adult    | 0.860     | 0.873     | **0.874**                                   |
| covtype  | 0.832     | **0.835** | 0.826                                       |
| dna      | **0.950** | 0.949     | 0.946                                       |
| glass    | 0.766     | **0.813** | 0.719                                       |
| mandelon | **0.882** | 0.881     | 0.866                                       |
| soybean  | **0.936** | **0.936** | 0.917                                       |
| yeast    | **0.591** | 0.573     | 0.542                                       |

## Implementation

To understand the implementation of `TreeGrad`, we interpret a decision tree algorithm to be a three layer neural network, where the layers are as follows:

1. **Node layer** — determines the decision boundaries
2. **Routing layer** — determines which nodes route to final leaf nodes (global product routing)
3. **Leaf layer** — determines final predictions via fully connected dense layer

This approach is the same as Kontschieder, Peter, et al. "Deep neural decision forests."

## New in v2

- Migrated from `setup.py` / nose2 to `pyproject.toml` / pytest + uv
- Added PyTorch backend alongside autograd (see new notebooks)
- Ruff linting and formatting
- Deno fmt for markdown/config files
- Ty type checking
- New comparative notebooks: autograd vs. PyTorch implementations

## Citation

```bibtex
@inproceedings{siu2019transferring,
  title={Transferring Tree Ensembles to Neural Networks},
  author={Siu, Chapman},
  booktitle={International Conference on Neural Information Processing},
  pages={471--480},
  year={2019},
  organization={Springer}
}
```

Link: https://arxiv.org/abs/1904.11132
