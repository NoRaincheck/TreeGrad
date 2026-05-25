"""Shared fixtures for TreeGrad tests."""

import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def binary_X_y():
    """Binary classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_classes=2,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        n_features=10,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="session")
def multiclass_X_y():
    """Multiclass classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        n_features=10,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="session")
def regression_X_y():
    """Regression dataset."""
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    return X, y
