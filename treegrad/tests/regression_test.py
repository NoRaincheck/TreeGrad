"""Tests for regression with TGDRegressor."""

import numpy as np
from treegrad import TGDRegressor


def test_regressor_predict_shape(regression_X_y):
    X, y = regression_X_y
    model = TGDRegressor(autograd_config={"num_iters": 1})
    model.fit(X, y)
    assert model.predict(X).shape[0] == X.shape[0]


def test_regressor_partial_fit_changes_predictions(regression_X_y):
    X, y = regression_X_y
    model = TGDRegressor(autograd_config={"num_iters": 1})
    model.fit(X, y)
    a1 = model.predict(X)

    # partial fit off lightgbm
    model.partial_fit(X, y)
    a2 = model.predict(X)
    assert not np.array_equal(a1, a2)

    # partial fit off itself
    model.partial_fit(X, y)
    a3 = model.predict(X)
    assert not np.array_equal(a1, a3)
