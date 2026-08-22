import numpy as np
import torch

from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier

from treegrad import TGDClassifier
from treegrad.model import TorchTreeEnsemble
from treegrad.tree_utils import (
    multi_tree_to_param,
    multiclass_trees_to_param,
    split_trees_by_classes,
    gbm_gen,
)


def _binary_params():
    X, y = make_classification(
        120,
        n_classes=2,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        n_features=8,
        random_state=3,
    )
    m = LGBMClassifier(
        n_estimators=5, num_leaves=7, random_state=1, verbose=-1, num_threads=1
    ).fit(X, y)
    trees = [t["tree_structure"] for t in m.booster_.dump_model()["tree_info"]]
    return X, y, multi_tree_to_param(X, y, trees)


def _multiclass_params():
    X, y = make_classification(
        120,
        n_classes=3,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        n_features=8,
        random_state=4,
    )
    m = LGBMClassifier(
        n_estimators=4, num_leaves=7, random_state=1, verbose=-1, num_threads=1
    ).fit(X, y)
    trees = [t["tree_structure"] for t in m.booster_.dump_model()["tree_info"]]
    tp = multiclass_trees_to_param(X, y, split_trees_by_classes(trees, 3))
    return X, y, tp


def test_batched_forward_matches_legacy_binary():
    X, _, tp = _binary_params()
    legacy = gbm_gen(tp[0], X, tp[2], tp[1], False, 2)(tp[0], X).detach().numpy()
    batched = (
        TorchTreeEnsemble(tp[0], tp[2], tp[1], num_classes=2, dtype=torch.float64)(X)
        .detach()
        .numpy()
    )
    assert np.abs(legacy - batched).max() < 1e-9


def test_batched_forward_matches_legacy_multiclass():
    X, _, tp = _multiclass_params()
    legacy = gbm_gen(tp[0], X, tp[2], tp[1], True, 3)(tp[0], X).detach().numpy()
    logits = (
        TorchTreeEnsemble(tp[0], tp[2], tp[1], num_classes=3, dtype=torch.float64)(X)
        .detach()
        .numpy()
    )
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    assert np.abs(legacy - probs).max() < 1e-9


def test_estimator_dtype_and_device_config():
    X, y = make_classification(80, n_classes=2, n_features=6, random_state=0)
    model = TGDClassifier(
        n_estimators=10, autograd_config={"num_iters": 2, "dtype": "float64"}
    )
    model.fit(X, y)
    model.partial_fit(X, y)
    assert model.model_.coef.dtype == torch.float64
    assert model.model_.coef.device.type == "cpu"


def test_tau_annealing_updates_temperature():
    X, y = make_classification(80, n_classes=2, n_features=6, random_state=2)
    model = TGDClassifier(
        n_estimators=5,
        autograd_config={"num_iters": 5, "tau": 0.05, "tau_end": 0.01},
    )
    model.fit(X, y)
    model.partial_fit(X, y)
    assert abs(model.model_.tau.item() - 0.01) < 1e-9


def test_training_options_smoke():
    X, y = make_classification(80, n_classes=2, n_features=6, random_state=6)
    cfg = {
        "num_iters": 4,
        "step_size": 0.02,
        "lr_schedule": "cosine",
        "l1_reg": 0.01,
        "shuffle": True,
    }
    model = TGDClassifier(n_estimators=5, autograd_config=cfg)
    model.fit(X, y)
    model.partial_fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (80, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_compile_mode_runs():
    X, y = make_classification(80, n_classes=2, n_features=6, random_state=1)
    model = TGDClassifier(
        n_estimators=5, autograd_config={"num_iters": 3, "compile": True}
    )
    model.fit(X, y)
    model.partial_fit(X, y)
    assert model.predict_proba(X).shape == (80, 2)


def test_multiclass_proba_sums_to_one():
    X, y = make_classification(
        100, n_classes=3, n_informative=4, n_features=8, random_state=5
    )
    model = TGDClassifier(n_estimators=10, autograd_config={"num_iters": 3})
    model.fit(X, y)
    model.partial_fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
