# Code walkthrough — what every part of icarus does

Purpose: own the codebase (supervisor Direction 2). This traces the data from a
raw `.mat` file to a trained heat-flux model and back to a prediction, file by
file, with the *why* behind the non-obvious lines. Read it next to the source.

Data convention throughout: 3-D fields are `[ny, nx, nt]` (height, width, time).
Flattened "snapshot matrices" are `[n_pix, nt]` (a pixel per row, a frame per
column). Per-sample feature/target rows are **time-major**: row index
`= t * n_pix + p`. Getting that ordering right is the difference between a valid
temporal split and silent train/test leakage — see `extractor.py` below.

---

## 1. `data/loader.py` — get arrays off disk

`load(path, temperature_key, heatflux_key, timestep_key)` dispatches on file
suffix to `_load_npz` / `_load_mat` / `_load_hdf5`.

- **`_load_mat`** first tries `scipy.io.loadmat` (works for MATLAB v5). If the
  key isn't found (a v7.3 file is actually HDF5), it **falls back to
  `_load_hdf5`**. This is why the big `MODEL_~*.MAT` files load at all.
- **`_load_hdf5`** does the critical de-transpose: MATLAB stores an
  `[ny, nx, nz, nt]` array reversed as `[nt, nz, nx, ny]`, so the code does
  `T.transpose(3, 2, 1, 0)` to restore `[ny, nx, nz, nt]`, then `[:, :, 0, :]`
  to take **z-layer 0 = the heater surface**. 3-D arrays get `transpose(2,1,0)`.
- Returns `{"T", "q", "dt"}` with `T`, `q` as `[ny, nx, nt]`.
- **Assumption to remember:** z-layer 0 is the surface. If your data's surface
  is elsewhere, slice before passing in.

## 2. `data/preprocessor.py` — crop, trim, mean-centre, reshape

`Preprocessor(PreprocessorConfig(spatial_crop, trim_frames)).fit_transform(data)`:

- **Spatial crop** removes `spatial_crop` pixels from each border (edge
  artefacts in IR). **Trim** drops the first `trim_frames` frames (startup
  transient).
- Computes the **per-pixel time-mean field** `_q_mean = q.mean(axis=2,
  keepdims=True)` → shape `[ny, nx, 1]`; the centred fields are `T_c = T - mean`,
  `q_c = q - mean`. POD/SPOD operate on the *centred* (fluctuation) field; the
  mean is added back only at the very end of prediction.
- `to_matrix(field)` = `field.reshape(ny*nx, nt)` → the `[n_pix, nt]` snapshot
  matrix POD wants. `from_matrix` inverts it.
- Returns `T, q, T_c, q_c` (+ stored mean for reconstruction).

## 3. `decomposition/pod.py` — POD via SVD

`POD(n_modes | energy_threshold).fit(X_c)`:

- Economy SVD `X_c = U Σ Vᵀ`. **Spatial modes** `modes_ = U[:, :r]`,
  **temporal coefficients** `temporal_coefficients_ = diag(S) @ Vt` (i.e.
  `A = Σ Vᵀ`).
- `energy_fractions_ = S² / ΣS²`, cumulative for "how many modes capture 99 %".
- **`modal_contributions(X_c)`** → `[n_pix, nt, n_modes]`, the rank-1 fields
  `C_i(x,t) = φ_i · σ_i · v_i(t)`. Implementation detail: it computes
  `A_in = modes_[:, :r].T @ X_c` (= `Σ Vᵀ` directly — **no extra σ
  multiplication**, a deliberate correctness rule) then fills per mode, with
  optional `chunk_size`/`dtype` to bound memory on big arrays. These
  contributions are the **features for Model C**.
- `project(X_c_new) = modes_.T @ X_c_new` gives coefficients on unseen data
  without refitting (used for test sets).

## 4. `decomposition/spod.py` — Spectral POD (new, summer Direction 1)

`SPOD(n_modes, block_size, overlap, window).fit(X_c, dt)`:

- **Welch segmentation:** split time into overlapping blocks of `block_size`
  frames (`step = block_size - overlap*block_size`); `n_blocks` of them.
- Each block is **Hann-windowed** then FFT'd along time (`np.fft.rfft`), scaled
  for energy consistency. Result `Q_hat[freq, pix, block]`.
- **Per frequency**, SVD of `Q_hat[k]` (`[n_pix, n_blocks]`): left singular
  vectors = **SPOD modes** at that frequency, singular values² = **modal
  energies** (eigenvalues of the cross-spectral density).
- `spectrum(mode)` = energy of a mode rank vs frequency (peaks = dominant
  timescales). `mode(freq)` = the coherent spatial structure at a frequency.
- Unlike POD (modes mix all frequencies), each SPOD mode lives at **one
  frequency** — the basis for separating nucleation / departure / microlayer
  by timescale.

## 5. `decomposition/dmd.py` — DMD forecasting

Dynamic Mode Decomposition: fits a linear operator `A` s.t. `x_{k+1} ≈ A x_k`
in a reduced POD subspace, giving eigenvalues (growth/decay + frequency) and
modes. `forecast_from(x0, n_steps)` rolls the dynamics forward. Suited to
**short-horizon** prediction; accuracy decays over long horizons.

## 6. `features/engineer.py` — build model inputs/targets

- **`build_raw_features(T)`** (Model A): temperature only, flattened
  **time-major** via `transpose(2,0,1).reshape(...)`.
- **`build_gradient_features(T)`** (Model B): stacks `T, dT/dt, dT/dx, dT/dy`.
- **`build_modal_features(...)`** (Model C): the POD modal contributions.
- **`flatten_target(q) = q.transpose(2,0,1).reshape(-1)`** — time-major target.
- **`train_test_split_temporal(X, y, train_fraction, n_pix, nt)`** splits on the
  `nt_train * n_pix` row boundary — valid **only because** rows are time-major
  (the first block of rows = the first timesteps).

## 7. `models/neural.py` — the MLP and its tuning

`HeatFluxNet(strategy, hidden_layer_sizes, ...)` wraps an sklearn `MLPRegressor`
with internal `StandardScaler`s for X and y (scaling inside `fit`/`optimise`
to avoid leakage).

- **`optimise(...)`** runs Optuna Bayesian search. Subsampling for speed goes
  through `_subsample_indices(...)`, which **sorts the indices when
  `validation_strategy="temporal"`** so the 80/20 head/tail split is genuinely
  chronological (a fixed bug: a random subsample had made "temporal" actually
  random).
- `SEARCH_SPACES` presets: `small | medium | large`.
- `fit` then `predict`. For `strategy="modal"`, predict returns per-mode
  contributions that get summed downstream.

## 8. `metrics/evaluation.py` — scoring

`evaluate(y_true, y_pred, split)` → `Metrics(r2, rmse, mae, split)`. RMSE/MAE in
W/m². Note the **fluctuation vs absolute** distinction is applied by the caller
(trainer), not here — this just computes the numbers on whatever it's given.

## 9. `pipeline/runner.py` — single-dataset end-to-end

`Pipeline(strategy, ...).fit(data)` chains §2→§3→§6→§7:
preprocess → POD → build features → temporal split → (optionally optimise) →
train. Then:

- **`evaluate()`** scores train/test.
- **`predict(T_new)`** → `_predict_modal` for Model C: predicts modal
  contributions, **sums them** (`q_modal_pred.sum(axis=1)`), then **adds the
  mean field back** via tiling `q_mean_vec` across time. Correctness rule:
  *do not* re-multiply predictions by `φ_i` — sum contributions directly.
- Fixed bug: `fit()` now creates a model only `if self.model_ is None`, so a
  manually injected `HeatFluxNet` (with preset hyperparams) is respected.

## 10. `registry/` — the multi-dataset machinery

- **`dataset.py`** — `DatasetRegistry` + `DatasetEntry`: a JSON-indexed store of
  datasets (raw/processed paths, per-dataset params, status). `register`,
  `get`, `list_datasets`, `summary`.
- **`extractor.py`** — `FeatureExtractor.process(ds_id)`: load → preprocess →
  fit POD on the **train fraction** → compute modal contributions → write
  `features.h5`. **Critical ordering line:** contributions are reshaped
  **time-major** (`transpose(1,0,2)` before reshape) so the stored `split`
  mask (`first nt_train*n_pix rows = train`) is a real temporal split. It also
  stores `q_flat`, **`q_abs`** (absolute target, time-major, row-aligned), and
  **`q_mean_field`** (per-pixel mean) for absolute-R² reconstruction.
- **`trainer.py`** — `MultiDatasetTrainer`:
  - `fit(dataset_ids)` pools each dataset's internal train/test split.
  - **`cross_dataset_fit(train_ids, test_id)`** — the headline: trains on *all*
    rows of the train datasets, holds out *all* rows of `test_id` (raises if
    `test_id` is in `train_ids`). This is the leave-one-surface-out protocol.
  - **`evaluate()`** reports **fluctuation** R² (`"test"`, the meaningful
    metric — on the mean-subtracted field the model actually predicts) **and**
    **absolute** R² (`"test_absolute"`, mean added back; always higher because
    the spatial mean inflates variance). Skips absolute gracefully if `q_abs`
    is missing (old feature files).

## 11. `visualisation/plots.py`

Publication figures: `plot_field`, `plot_pod_modes`, `plot_cumulative_energy`,
`plot_scatter`, `plot_model_summary` (6-panel eval). Least-tested module.

---

## The two end-to-end paths, in one line each

- **Single dataset:** `load → Preprocessor → POD → build_modal_features →
  train_test_split_temporal → HeatFluxNet.optimise/fit → evaluate/predict`
  (`Pipeline`).
- **Cross-dataset:** `register 3 datasets → FeatureExtractor.process each
  (writes features.h5) → MultiDatasetTrainer.cross_dataset_fit(train, holdout)
  → evaluate` (fluctuation + absolute R²).

## Where the summer directions plug in

- **SPOD (§4)** — done as a decomposition primitive; next: an extractor/feature
  path that uses SPOD modes instead of POD modes, and a frequency-content
  analysis script.
- **Heat partition** — likely a new target-construction step (per-mechanism
  `q` components) in `features/` + a multi-output head in `models/`.
- **Stochastic** — a probabilistic variant of `HeatFluxNet` (ensemble / quantile
  / MDN) and calibration metrics in `metrics/`.
