# icarus

**Data-driven heat flux prediction from infrared thermography.**

`icarus` provides a full pipeline from raw IR camera data to trained
heat flux prediction models using Proper Orthogonal Decomposition (POD),
Dynamic Mode Decomposition (DMD), and artificial neural networks.

It implements the methodology from:

> *Investigating the efficacy of data-driven techniques and machine learning
> algorithms to predict heat transfer characteristics* (Twum-Barima, 2025)

The best-performing approach (Model C: POD modal mapping) achieved **R² = 0.729**
on a 17M-sample flow boiling dataset — a 69 % improvement over the linear baseline.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/twumbarimaraymond1-coder/icarus
```

Or from source (recommended for development):

```bash
git clone https://github.com/twumbarimaraymond1-coder/icarus
cd icarus
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, NumPy, SciPy, scikit-learn, Optuna, Matplotlib.
The package name on install is `icarus-thermal`; the import name is `icarus`.

---

## Quickstart

```python
import icarus as tf

# Load your dataset (.mat, .h5, .npz supported)
data = tf.data.loader.load(
    "experiment.mat",
    temperature_key="T",
    heatflux_key="qL2",
)

# Or load from numpy arrays directly
import numpy as np
data = tf.data.loader.from_arrays(T, q, dt=2.5e-4)

# Run the full pipeline (POD modal strategy, best performance)
pipeline = tf.Pipeline(
    strategy="modal",   # "raw" | "gradient" | "modal"
    n_pod_modes=5,
    spatial_crop=5,
    trim_frames=43,
    optimise_hyperparams=True,
    n_trials=30,
)
pipeline.fit(data)

# Evaluate
metrics = pipeline.evaluate()
# [test]  R² = 0.7293  RMSE = 25,959 W/m²  MAE = 20,656 W/m²

# Predict on new data
q_predicted = pipeline.predict(T_new)   # shape [ny, nx, nt]
```

---

## Three model strategies

| Strategy | Features | Notes |
|----------|----------|-------|
| `"raw"` (Model A) | Temperature only | Baseline |
| `"gradient"` (Model B) | T + dT/dt + dT/dx + dT/dy | Modest improvement |
| `"modal"` (Model C) | POD modal contributions | **Best: R² = 0.729** |

The modal strategy works by:
1. Decomposing the temperature field into dominant POD modes
2. Learning a mapping from temperature modal coefficients → heat flux modal coefficients
3. Reconstructing the full heat flux field from the predicted coefficients

---

## Cross-dataset generalisation (multi-dataset workflow)

Beyond the single-dataset `Pipeline`, icarus includes a dataset registry for
training across multiple experiments and testing on a fully held-out one
(leave-one-surface-out). This answers the stronger question *"does the
temperature→heat-flux modal coupling transfer to an experiment the model has
never seen?"* — not merely to unseen timesteps of the same experiment.

```python
from icarus.registry.dataset import DatasetRegistry, DatasetEntry
from icarus.registry.extractor import FeatureExtractor
from icarus.registry.trainer import MultiDatasetTrainer

reg = DatasetRegistry("~/.icarus/datasets")
for ds_id, path in [("D001", "surface1.mat"),
                    ("D002", "surface2.mat"),
                    ("D003", "surface3.mat")]:
    reg.register(DatasetEntry(ds_id, "water", "flow_boiling", "patch",
                              "MyLab", raw_path=path))

ext = FeatureExtractor(reg, n_pod_modes=5)
for ds_id in ("D001", "D002", "D003"):
    ext.process(ds_id)

trainer = MultiDatasetTrainer(reg, n_pod_modes=5)
trainer.cross_dataset_fit(train_ids=["D001", "D002"], test_id="D003")
metrics = trainer.evaluate()   # fluctuation + absolute-field R²/RMSE
```

A runnable end-to-end script is provided in
[`examples/cross_dataset_real.py`](examples/cross_dataset_real.py)
(real `.mat` files) and
[`examples/cross_dataset_eval.py`](examples/cross_dataset_eval.py) (synthetic).

---

## Spectral POD (frequency-resolved structures)

Where POD ranks modes by energy alone, **SPOD** produces modes coherent at a
single frequency — separating boiling structures by timescale (nucleation,
bubble departure, microlayer). The high-level API mirrors the Quickstart:

```python
import icarus as tf

# Low-memory load of one field (reads only the heater layer of 4-D temperature)
field, dt = tf.load_field("MODEL_~1.MAT", field="heatflux")

# Fit Spectral POD straight from the [ny, nx, nt] field
spod = tf.SPOD(block_size=1024).fit_field(field, dt=dt, spatial_crop=5, trim_frames=43)

print(spod.dominant_frequencies(n=4))       # candidate dominant timescales (Hz)
spod.plot_spectrum("spectrum.png")          # energy vs frequency
spod.plot_mode(spod.dominant_frequencies()[0], "mode.png")  # structure at top peak
```

A runnable version is in [`examples/spod_analysis.py`](examples/spod_analysis.py).

---

## Timescale-resolved prediction with uncertainty

`BandwiseModalModel` splits temperature and heat flux into frequency bands and
learns a separate Model-C mapping per band ("Model C per timescale"), so you
see which timescales transfer. With `n_members > 1` each band is a deep
ensemble, giving a **calibrated uncertainty band** — combining epistemic
(model disagreement) and aleatoric (irreducible boiling randomness) terms.

```python
import icarus as tf

T, _  = tf.load_field("MODEL_~1.MAT", "temp")
q, dt = tf.load_field("MODEL_~1.MAT", "heatflux")

model = tf.BandwiseModalModel(edges=[200, 1000], n_pod_modes=5, n_members=5)
model.fit(T, q, dt=dt, spatial_crop=5, trim_frames=43)

model.evaluate()                                   # total + per-band R²
mean, lower, upper = model.predict_interval(T, coverage=0.9)
tf.interval_metrics(q[5:-5, 5:-5, 43:], lower, upper)   # coverage + sharpness
```

Runnable: [`examples/bandwise_stochastic.py`](examples/bandwise_stochastic.py).

---

## Metrics: fluctuation vs absolute R²

`MultiDatasetTrainer.evaluate()` reports **two** test metrics, and the
distinction matters when comparing results:

- **Fluctuation R²** (returned as `"test"` / `"test_fluctuation"`) is computed
  on the mean-subtracted heat-flux field — the quantity the POD modal model
  actually predicts. It measures how well the temperature→heat-flux *modal
  coupling* is captured. This is the honest headline number.
- **Absolute R²** (`"test_absolute"`) adds the per-pixel time-mean field back
  to both truth and prediction. It is always more flattering, because the
  large quasi-static spatial mean dominates the variance.

When citing or comparing results from this package, state which metric you
are using.

---

## Assumptions & conventions (read before using your own data)

- **Array convention** is `[ny, nx, nt]` for all 3-D fields. Time-major
  flattening (`transpose(2, 0, 1)` before reshape) is used throughout so that
  temporal train/test splits are genuine past→future splits.
- **4-D temperature arrays** `[ny, nx, nz, nt]` are reduced by taking
  **z-layer 0**, assumed to be the heater surface. If your surface is at a
  different layer, slice before loading (`from_arrays(T[:, :, k, :], q)`).
- **MATLAB v7.3 files** (HDF5-based) are handled automatically, including
  MATLAB's reversed axis storage order.
- **Default variable names** are `T` (temperature), `qL2` (heat flux), and
  `TimeStep` (scalar dt in seconds) — all overridable via keyword arguments
  to `load()` / `FeatureExtractor.process()`.
- **Units** are assumed to be kelvin and W/m²; RMSE/MAE are reported in the
  units of the heat-flux input.
- POD `modal_contributions()` returns `U^T X_c` scaling (no extra σ
  multiplication); see docstrings before composing with your own SVD code.

---

## Individual components

You can also use the modules independently:

```python
from icarus.decomposition.pod import POD
from icarus.data.preprocessor import Preprocessor

# Preprocessing
pre = Preprocessor()
out = pre.fit_transform(data)
X_c = Preprocessor.to_matrix(out["T_c"])   # [n_pix, nt]

# POD
pod = POD(n_modes=10)
pod.fit(X_c)
print(f"First 5 modes capture {pod.cumulative_energy_[4]:.1%} of variance")

# Modal contributions
contribs = pod.modal_contributions(X_c)    # [n_pix, nt, n_modes]

# Visualisation
from icarus.visualisation.plots import plot_pod_modes, plot_cumulative_energy
ny, nx = out["T"].shape[:2]
plot_cumulative_energy(pod)
plot_pod_modes(pod, ny=ny, nx=nx, n_modes=5)
```

```python
from icarus.decomposition.dmd import DMD

# DMD forecasting
dmd = DMD(energy_threshold=0.99, dt=2.5e-4)
dmd.fit(X_c_train)
X_forecast = dmd.forecast_from(X_c_train[:, -1], n_steps=1200)
```

---

## Visualisation

```python
from icarus.visualisation.plots import (
    plot_field,
    plot_pod_modes,
    plot_cumulative_energy,
    plot_scatter,
    plot_model_summary,
)

# Single field
plot_field(q[:, :, 100], title="Heat flux at t=100")

# Full model evaluation summary (6-panel figure)
plot_model_summary(
    q_true_field, q_pred_field,
    y_true_flat, y_pred_flat,
    metrics_train, metrics_test,
    r2_t=r2_t, rmse_t=rmse_t,
    model_name="Model C — POD Modal",
)
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Project structure

```
icarus/
├── data/
│   ├── loader.py          # .mat, .h5, .npz, numpy array loading
│   └── preprocessor.py    # cropping, mean-centering, reshaping
├── decomposition/
│   ├── pod.py             # POD via SVD
│   └── dmd.py             # DMD forecasting
├── features/
│   └── engineer.py        # gradient and modal feature construction
├── models/
│   └── neural.py          # MLP with Bayesian optimisation
├── metrics/
│   └── evaluation.py      # R², RMSE, MAE
├── visualisation/
│   └── plots.py           # spatial fields, modes, diagnostics
└── pipeline/
    └── runner.py          # end-to-end Pipeline
```

---

## Data availability

The flow-boiling experimental datasets used to develop and validate this
package were produced at Loughborough University and are **not redistributed
in this repository**; they may be available from the authors / Loughborough
University on reasonable request. All code paths can be exercised without
them: `examples/quickstart.py` and `examples/cross_dataset_eval.py` generate
synthetic data, and the test suite (`pytest tests/`) is fully self-contained.

---

## Citation

If you use icarus in academic work, please cite it (see `CITATION.cff`):

> Twum-Barima, R. (2026). *icarus: data-driven heat flux prediction from
> infrared thermography* (v0.1.0) [Computer software].
> https://github.com/twumbarimaraymond1-coder/icarus

---

## Known limitations

- Experimental datasets are not included in this repository (see *Data
  availability*).
- The reported Model C R² = 0.729 is dataset-specific and should be revalidated on independent datasets before being cited as a general result.
- The default ANN search space (`"medium"`) is designed for moderate-sized datasets with 5 POD modes. Larger mode counts or datasets may require `hyperparam_search_space="large"` and more Optuna trials.
- Current models use scikit-learn MLPs. Future versions may include PyTorch models for larger-scale training and GPU acceleration.
- DMD forecasting accuracy degrades over longer horizons — it is suited to short-horizon prediction only.

## Contributing

Contributions welcome — particularly additional datasets, fluid-specific
pre-trained models, and improved DMD variants. See `CONTRIBUTING.md`.

## Licence

MIT
