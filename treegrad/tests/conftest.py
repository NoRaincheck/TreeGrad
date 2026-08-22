import treegrad  # noqa: F401

# Importing treegrad (and therefore torch) before any test module can pull
# in lightgbm avoids the macOS duplicate-OpenMP-runtime segfault.
