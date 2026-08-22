"""
Tree Grad

Implementation of an online learning approach for tree based models

The differentiable ensemble runs on a batched torch implementation
(``treegrad.model.TorchTreeEnsemble``): all trees are padded and stacked
into tensors, trained with ``torch.optim.Adam`` using numerically stable
losses (BCEWithLogits / cross-entropy / MSE / Huber).
"""
import sys

import lightgbm as lgb
import numpy as np
import torch
from scipy.special import expit

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from treegrad.model import (
    TorchTreeEnsemble,
    fit_ensemble,
    make_loss_fn,
    to_numpy,
)
from treegrad.tree_utils import (
    split_trees_by_classes,
    multi_tree_to_param,
    multiclass_trees_to_param,
)


def _dump_trees(base_model):
    model_dump = base_model.booster_.dump_model()
    return [m["tree_structure"] for m in model_dump["tree_info"]]


def _softmax(scores):
    z = scores - scores.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


class BaseTreeGrad(BaseEstimator):
    _task = None

    def __init__(
        self,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        autograd_config={"refit_splits": False, "batch_size": 32},
    ):
        self.ensemble_config = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
        }
        self.autograd_config = autograd_config

    def _cfg(self, key, default=None):
        return self.autograd_config.get(key, default)

    def _lgb_params(self):
        params = dict(self.ensemble_config)
        if sys.platform == "darwin":
            # avoid OpenMP runtime clash between lightgbm and torch on macOS
            params.setdefault("num_threads", 1)
        return params

    def _torch_kwargs(self):
        return {
            "dtype": self._cfg("dtype", torch.float32),
            "device": self._cfg("device", "cpu"),
            "tau": self._cfg("tau", 0.01),
        }

    def _train_options(self):
        return {
            "step_size": self._cfg("step_size", 0.05),
            "num_iters": self._cfg("num_iters", 1000),
            "batch_size": self._cfg("batch_size", 32),
            "tau_end": self._cfg("tau_end"),
            "shuffle": self._cfg("shuffle", True),
            "lr_schedule": self._cfg("lr_schedule"),
            "l1_reg": self._cfg("l1_reg", 0.0),
            "compile_mode": self._cfg("compile", False),
            "verbose": self._cfg("verbose", False),
        }

    def _trees_params(self, X, y):
        trees_ = _dump_trees(self.base_model_)
        if self._task == "classification" and self.n_classes_ > 2:
            trees = split_trees_by_classes(trees_, self.n_classes_)
            return multiclass_trees_to_param(X, y, trees)
        return multi_tree_to_param(X, y, trees_)

    def _make_model(self, trees_params):
        num_classes = self.n_classes_ if self._task == "classification" else 1
        return TorchTreeEnsemble(
            trees_params[0],
            trees_params[2],
            trees_params[1],
            num_classes=num_classes,
            **self._torch_kwargs(),
        )

    def _make_loss_fn(self):
        return make_loss_fn(
            self._task,
            num_classes=self.n_classes_,
            loss=self._cfg("loss", "mse"),
            l1_reg=self._cfg("l1_reg", 0.0),
        )

    def _fit_torch(self, X, y):
        trees_params = self._trees_params(X, y)
        self.base_param_ = trees_params
        model = self._make_model(trees_params)
        fit_ensemble(model, X, y, self._make_loss_fn(), **self._train_options())
        self.model_ = model
        self.is_partial = True

    def _continue_torch(self, X, y):
        # warm start: continue optimising the existing ensemble parameters
        fit_ensemble(self.model_, X, y, self._make_loss_fn(), **self._train_options())
        self.is_partial = True

    def _scores(self, X):
        model = self.model_
        with torch.no_grad():
            X_t = torch.as_tensor(
                np.asarray(X), dtype=model.coef.dtype, device=model.coef.device
            )
            return to_numpy(model(X_t))


class TGDClassifier(BaseTreeGrad, ClassifierMixin):
    _task = "classification"

    def fit(self, X, y):
        self.base_model_ = lgb.LGBMClassifier(**self._lgb_params())
        self.base_model_.fit(np.asarray(X), np.asarray(y))
        self.n_classes_ = self.base_model_.n_classes_
        self.classes_ = self.base_model_.classes_
        self.is_partial = False
        return self

    def partial_fit_base(self, X, y):
        check_is_fitted(self, "base_model_")
        X = np.asarray(X)
        y = np.asarray(y)
        self._fit_torch(X, y)
        return self

    def partial_fit_param(self, X, y):
        check_is_fitted(self, "base_model_")
        check_is_fitted(self, "base_param_")
        check_is_fitted(self, "model_")
        self._continue_torch(np.asarray(X), np.asarray(y))
        return self

    def partial_fit(self, X, y):
        check_is_fitted(self, "base_model_")
        if self.is_partial:
            self.partial_fit_param(X, y)
        else:
            self.partial_fit_base(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "base_model_")
        if not self.is_partial:
            return self.base_model_.predict(X)
        scores = self._scores(X)
        if self.n_classes_ > 2:
            return np.argmax(scores, axis=1)
        return np.round(expit(scores))

    def predict_proba(self, X):
        check_is_fitted(self, "base_model_")
        if not self.is_partial:
            return self.base_model_.predict_proba(X)
        scores = self._scores(X)
        if self.n_classes_ > 2:
            return _softmax(scores)
        pred_positive = expit(scores)
        return np.stack([1 - pred_positive, pred_positive], axis=-1)


class TGDRegressor(BaseTreeGrad, RegressorMixin):
    _task = "regression"

    def fit(self, X, y):
        self.base_model_ = lgb.LGBMRegressor(**self._lgb_params())
        self.base_model_.fit(np.asarray(X), np.asarray(y))
        self.n_classes_ = 1
        self.is_partial = False
        return self

    def partial_fit_base(self, X, y):
        check_is_fitted(self, "base_model_")
        self._fit_torch(np.asarray(X), np.asarray(y))
        return self

    def partial_fit_param(self, X, y):
        check_is_fitted(self, "base_model_")
        check_is_fitted(self, "base_param_")
        check_is_fitted(self, "model_")
        self._continue_torch(np.asarray(X), np.asarray(y))
        return self

    def partial_fit(self, X, y):
        check_is_fitted(self, "base_model_")
        if self.is_partial:
            self.partial_fit_param(X, y)
        else:
            self.partial_fit_base(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "base_model_")
        if not self.is_partial:
            return self.base_model_.predict(X)
        return self._scores(X)


if __name__ == "__main__":
    # these are test cases - to be refactored out.
    from sklearn.datasets import make_classification

    X, y = make_classification(
        100,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        n_features=10,
    )
    model = TGDClassifier(autograd_config={"num_iters": 5})
    model.fit(X, y)
    print(model.predict(X))

    # partial fit off lightgbm
    model.partial_fit(X, y)
    print(model.predict(X))

    # partial fit off itself
    model.partial_fit(X, y)
    print(model.predict(X))

    # test class binary
    X, y = make_classification(
        100,
        n_classes=2,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        n_features=8,
    )
    model = TGDClassifier(autograd_config={"num_iters": 100})
    model.fit(X, y)
    print(model.predict(X))
    print(np.round(model.predict_proba(X)))

    # partial fit off lightgbm
    model.partial_fit(X, y)
    print(model.predict(X))
    print(np.round(model.predict_proba(X)))

    # partial fit off itself
    model.partial_fit(X, y)
    print(model.predict(X))
