from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
import numpy as np
from treegrad import TGDRegressor


def test_binary():
    # test class binary
    X, y = make_regression()
    model = TGDRegressor(autograd_config={"num_iters": 1})
    model.fit(X, y)
    a1 = model.predict(X)
    assert a1.shape[0] == X.shape[0]

    # partial fit off lightgbm
    model.partial_fit(X, y)
    a2 = model.predict(X)
    assert a2.shape[0] == X.shape[0]

    # partial fit off itself
    model.partial_fit(X, y)
    a3 = model.predict(X)
    assert a3.shape[0] == X.shape[0]

    assert not np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)


def test_regression_quality_and_continuity():
    X, y = make_regression(300, n_features=8, n_informative=6, noise=5.0, random_state=7)
    model = TGDRegressor(
        n_estimators=20, autograd_config={"num_iters": 150, "batch_size": 64}
    )
    model.fit(X, y)
    model.partial_fit(X, y)
    tuned = model.predict(X)

    assert np.isfinite(tuned).all()
    assert r2_score(y, tuned) > 0.6

    # regression outputs must be continuous, not rounded class-like labels
    unique_ratio = len(np.unique(np.round(tuned, 6))) / len(tuned)
    assert unique_ratio > 0.9


def test_regression_huber_loss():
    X, y = make_regression(150, n_features=6, n_informative=5, random_state=11)
    model = TGDRegressor(
        n_estimators=10, autograd_config={"num_iters": 30, "loss": "huber"}
    )
    model.fit(X, y)
    model.partial_fit(X, y)
    model.partial_fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (150,)
    assert np.isfinite(preds).all()
