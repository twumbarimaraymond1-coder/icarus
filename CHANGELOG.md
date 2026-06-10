# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/).

## [0.1.0] — 2026-06-10

First tagged release.

### Added
- Full single-dataset `Pipeline` (raw / gradient / modal strategies) with
  POD, DMD, MLP models and Optuna Bayesian hyperparameter optimisation.
- Dataset registry (`DatasetRegistry`, `FeatureExtractor`,
  `MultiDatasetTrainer`) for multi-dataset training.
- **Cross-dataset generalisation**: `MultiDatasetTrainer.cross_dataset_fit()`
  trains on N datasets and holds out one entirely (leave-one-surface-out),
  with a leakage guard. Runner script `examples/cross_dataset_real.py`.
- Dual test metrics: fluctuation-field R² (headline) and absolute-field R²
  (`test_absolute`), with `q_abs` / `q_mean_field` stored per dataset.
- MATLAB v7.3 (HDF5) loading with automatic axis de-transposition and
  heater-surface (z-layer 0) extraction, with regression tests.
- Memory-bounded POD modal contributions (`chunk_size`, `dtype` parameters).
- CI: pytest matrix (3.9–3.11), ruff lint job, quickstart smoke test.
- `CITATION.cff`.

### Fixed
- **Extractor split-mask ordering**: feature rows are now stored time-major,
  making the registry's internal train/test split genuinely temporal.
  ⚠ `features.h5` files extracted with earlier code are incompatible —
  re-run `FeatureExtractor.process(..., force=True)`.
- `Pipeline.fit()` no longer overwrites a manually injected `HeatFluxNet`.
- `HeatFluxNet.optimise(validation_strategy="temporal")` now sorts the
  subsample so the 80/20 validation split is truly chronological.
- POD `modal_contributions` / `project` use `U^T X_c` directly (no double
  singular-value scaling); time-major flattening enforced in all feature
  builders (both guarded by tests).
