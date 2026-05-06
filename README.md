# icarus

![CI](https://github.com/twumbarimaraymond1-coder/icarus/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/badge/PyPI-icarus--thermal-orange)

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

```bash
pip install icarus-thermal

> **Note:** `icarus` on PyPI is an unrelated astrophysics package. Install as `icarus-thermal`.
```

Or from source:

```bash
git clone https://github.com/twumbarimaraymond1-coder/icarus
cd icarus
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, NumPy, SciPy, scikit-learn, Optuna, Matplotlib

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

## Contributing

Contributions welcome — particularly additional datasets, fluid-specific
pre-trained models, and improved DMD variants. See `CONTRIBUTING.md`.

## Licence

MIT

---

## Research background

Icarus implements the methodology from:

> Twum-Barima, R. (2025). *Investigating the efficacy of data-driven techniques 
> and machine learning algorithms to predict heat transfer characteristics.*
> Individual Project Report, Loughborough University.

The best-performing approach (Model C: POD modal mapping) achieved **R² = 0.729** 
on 17,384,000 pixel-time samples from a flow boiling infrared thermography dataset,
representing a 69% improvement over the linear temperature baseline.

The key finding motivating Model C is that temperature and heat flux POD temporal 
coefficients are highly correlated (r ≥ 0.96 for dominant modes), providing physical 
justification for learning a reduced-order mapping between the two fields.

## Data

This repository contains no experimental data. The library operates on data you 
supply. See [DATA.md](DATA.md) for contribution guidelines and the dataset registry.
