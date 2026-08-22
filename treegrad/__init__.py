"""TreeGrad."""
import sys
import warnings

# On macOS, importing lightgbm before torch leads to duplicate OpenMP
# runtime crashes (segfaults inside lightgbm fits or torch ops). Importing
# torch first claims the runtime, and TreeGrad estimators keep lightgbm
# single-threaded on darwin to stay safe in every import order.
import torch  # noqa: F401

if "lightgbm" in sys.modules:
    warnings.warn(
        "lightgbm was imported before treegrad; on macOS this combination "
        "can segfault due to duplicate OpenMP runtimes. Import treegrad "
        "(or torch) before lightgbm.",
        stacklevel=2,
    )

from treegrad.treegrad import TGDClassifier, TGDRegressor

__version__ = "1.1.0"
