# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/).

## [0.3.0] — 2026-06-17

### Changed
- Refactored `SPOD` to match the `POD` structure: `fit()` is now a clean
  sequence over `_welch_transform()` and `_decompose_per_frequency()` helpers
  (behaviour unchanged, guarded by `TestSPOD`).

### Added
- `SPOD.dominant_frequencies(n)` — energy-sorted peak frequencies. The
  `examples/spod_analysis.py` runner now uses it instead of a duplicated
  local peak-finder.
- v0.2.0 release notes (`docs/RELEASE_NOTES_v0.2.0.md`).

## [0.2.0] — 2026-06-17

### Added
- **Spectral POD** (`icarus.decomposition.spod.SPOD`) via Welch's method —
  frequency-coherent modes that separate structures by timescale. Guarded by
  `TestSPOD` (including a two-tone separation test).
- `examples/spod_analysis.py` — diagnostic SPOD runner (energy spectrum +
  leading-mode maps), with a low-memory loader that reads only the heater
  z-layer off disk.
- `docs/literature_survey.md` (heat-partition + probabilistic/UQ prediction)
  and `docs/code_walkthrough.md` (file-by-file Model C / pipeline walkthrough).

### Fixed
- `Pipeline._predict_modal`: Model C inference on new data now sums modal
  contributions (which already include phi_i) and reshapes time-major, instead
  of double-counting phi_i and scrambling pixel/time axes. `predict()` was
  returning garbage while `evaluate()` was correct.
- Extractor stores features time-major so the internal train/test split is
  genuinely temporal (was a spatial pixel split).
- `Pipeline.fit()` respects a manually injected `HeatFluxNet`.

Test count: 32 → 54 passing.

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
