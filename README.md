# TreeGrad

![PyPI Version](https://img.shields.io/pypi/v/treegrad)

`TreeGrad` implements a naive approach to converting a Gradient Boosted Tree Model to an Online trainable model. It does this by creating differentiable tree models which can be learned via auto-differentiable frameworks. `TreeGrad` is in essence an implementation of Kontschieder, Peter, et al. "Deep neural decision forests." with extensions.

To install

```
uv sync
```

or alternatively from pypi


```
pip install treegrad
```

Run tests:

```
uv run pytest
```

```
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


# Usage

`fit` trains a LightGBM ensemble; `partial_fit` transfers those trees into a
differentiable `torch` model and fine-tunes them. Call `partial_fit`
repeatedly for continued online training.

```py
import treegrad as tgd

mod = tgd.TGDClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)
mod.fit(X, y)
mod.partial_fit(X, y)
```

Regression is available via `TGDRegressor`, which predicts continuous values.

## Training & performance options

The differentiable ensemble runs on a batched `torch` backend
(`treegrad.model.TorchTreeEnsemble`): all trees are stacked into padded
tensors so the forward pass is a handful of fused ops instead of a Python
loop over trees. Options are passed via the estimator's config dict:

```py
import torch

mod = tgd.TGDClassifier(autograd_config={
    "step_size": 0.05,      # Adam learning rate
    "num_iters": 1000,      # optimisation steps
    "batch_size": 32,       # mini-batch size
    "shuffle": True,        # reshuffle batches each pass
    "tau": 0.05,            # initial routing temperature
    "tau_end": 0.01,        # linear annealing target (None = fixed tau)
    "lr_schedule": "cosine",  # or None
    "l1_reg": 0.0,          # L1 penalty on split weights/biases
    "device": "cpu",        # "cuda" / "mps" for GPU acceleration
    "dtype": torch.float32,   # or torch.float64
    "compile": False,       # opt-in torch.compile (eager fallback)
})
```

Regression (`TGDRegressor`) trains proper regression objectives on raw leaf
outputs - `"loss": "mse"` (default) or `"loss": "huber"` - and predicts
continuous values.

Note (macOS): importing `lightgbm` before `torch`/`treegrad` can segfault
due to duplicate OpenMP runtimes; import `treegrad` first. TreeGrad's own
estimators default LightGBM to single-threaded on macOS to avoid this.

# Requirements

The requirements for this package are:

*  lightgbm
*  scikit-learn
*  pytorch

Future plans:

*  Add implementation for Neural Architecture search for decision boundary splits (requires a bit of clean up - TBA)
   *  Implementation can be done quite trivially using objects residing in `tree_utils.py` - Challenge is getting this working in a sane manner with `scikit-learn` interface.
*  GPU enabled auto differentiation framework - the model has been ported to `torch`, enabling GPU acceleration
*  support xgboost/lightgbm additional features such as monotone constraints
*  closed-form (NDF-style) leaf updates as an alternative to SGD-only leaf tuning

# Results

When decision splits are reset and subsequently re-learned, TreeGrad can be competitive in performance with popular implementations (albeit an order of magnitude slower). Below is a table showing accuracy on test dataset on UCI benchmark datasets for Boosted Ensemble models (100 trees)


| Dataset  | TreeGrad  | LightGBM  | Scikit-Learn (Gradient Boosting Classifier) |
| ---------| --------- | --------- | ------------------------------------------- |
| adult    | 0.860     | 0.873     | **0.874**                                   |
| covtype  | 0.832     | **0.835** | 0.826                                       |
| dna      | **0.950** | 0.949     | 0.946                                       |
| glass    | 0.766     | **0.813** | 0.719                                       |
| mandelon | **0.882** | 0.881     | 0.866                                       |
| soybean  | **0.936** | **0.936** | 0.917                                       |
| yeast    | **0.591** | 0.573     | 0.542                                       |


# Implementation

<!-- insert link to arxiv paper -->

## Tree as a neural network

`TreeGrad` interprets a decision tree as a three layer neural network:

1.  **Node layer**, which determines the decision boundaries. Axis-parallel splits are equivalent to a fully connected dense layer with one unit per split.
2.  **Routing layer**, which maps internal nodes to leaves via a binary routing matrix; the global product routing computes the probability of reaching each leaf.
3.  **Leaf layer**, which produces the final predictions from the routed probabilities.

This is the same formulation as Kontschieder, Peter, et al. "Deep neural decision forests." A LightGBM ensemble is first trained, then converted into this differentiable form (`tree_to_param` / `multi_tree_to_param`) so the split weights/biases and leaf values can be fine-tuned with gradient descent.

## Batched torch backend

The differentiable ensemble lives in `treegrad.model.TorchTreeEnsemble`. Rather than looping over individual trees in Python, all trees are padded to a common size and stacked into single tensors, so the entire forward pass runs as a few fused batched ops:

```
g(x)          = sigmoid(-clamp(decision / tau, -32, 32))   # soft routing gates
route_prob    = exp(log(g + eps) @ route.T)                # product routing
tree_output   = route_prob @ leaf_values
```

Padded ("fake") nodes and leaves contribute nothing because their routing matrix and leaf entries are zero, making the batched result numerically identical to the per-tree formulation (verified in tests to ~1e-16).

## Training

`partial_fit` optimises all parameters jointly with Adam over tensor-resident mini-batches (no per-step host transfers). Supported features:

*   Numerically stable objectives: `BCEWithLogits` for binary, `cross_entropy` for multiclass classification; proper regression objectives (`mse` or `huber`) on raw leaf outputs for `TGDRegressor`.
*   Optional L1 regularisation on split weights/biases.
*   Linear annealing of the routing temperature `tau -> tau_end`, which sharpens soft decisions towards hard splits during fine-tuning.
*   Optional cosine learning-rate schedule, mini-batch shuffling, and opt-in `torch.compile` (with automatic eager fallback).
*   Device/dtype selection (`cpu`, `cuda`, `mps`; `float32` or `float64`).

All of these are configured through the estimator's `autograd_config` dict — see [Training & performance options](#training--performance-options) above.

