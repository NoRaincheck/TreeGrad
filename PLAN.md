# TreeGrad — Development Plan

## Completed (v2.0)

- [x] Migrated from `setup.py` / `nose2.cfg` to `pyproject.toml` with `uv` for all tooling
- [x] Created `AGENTS.md` with ruff, ty, pytest, and deno formatting guidelines
- [x] Set major version bump to **2.0.0** (semantic versioning)
- [x] Added pytorch dependency group + re-implementation notebooks comparing autograd vs pytorch approaches
- [x] Fixed critical bug in incremental learning: `calculate_boundary` now caches coefficient/intercept values so decision boundaries stay consistent across batches with different feature ranges
- [x] Fixed catastrophic forgetting: `partial_fit_param` now replays original training data alongside new batches

## Remaining / Future Work

- [ ] Add integration tests for the pytorch re-implementation notebooks
- [ ] Benchmark autograd vs pytorch implementations on standard datasets
- [ ] Consider adding CI/CD with uv and pytest (GitHub Actions)
- [ ] Document incremental learning API more thoroughly in README
- [ ] Explore adding `partial_fit` support for `TGDRegressor` regression tasks
