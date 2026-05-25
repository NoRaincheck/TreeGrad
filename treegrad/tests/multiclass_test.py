"""Tests for multiclass classification with TGDClassifier."""

import numpy as np
from treegrad import TGDClassifier


def test_multiclass_predict_shape(multiclass_X_y):
    X, y = multiclass_X_y
    model = TGDClassifier(autograd_config={"num_iters": 1})
    model.fit(X, y)
    assert model.predict(X).shape[0] == X.shape[0]


def test_multiclass_predict_proba_shape(multiclass_X_y):
    X, y = multiclass_X_y
    model = TGDClassifier(autograd_config={"num_iters": 1})
    model.fit(X, y)
    a1 = model.predict_proba(X)
    assert a1.shape[1] == 3


def test_multiclass_partial_fit_changes_predictions(multiclass_X_y):
    X, y = multiclass_X_y
    model = TGDClassifier(autograd_config={"num_iters": 1})
    model.fit(X, y)
    a1 = model.predict_proba(X)

    # partial fit off lightgbm
    model.partial_fit(X, y)
    a2 = model.predict_proba(X)
    assert not np.array_equal(a1, a2)

    # partial fit off itself
    model.partial_fit(X, y)
    a3 = model.predict_proba(X)
    assert not np.array_equal(a1, a3)
