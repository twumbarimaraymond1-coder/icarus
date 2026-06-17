# icarus-thermal v0.2.0

**Install / upgrade:** `pip install -U icarus-thermal` · [PyPI](https://pypi.org/project/icarus-thermal/0.2.0/)

Second release. Adds Spectral POD, fixes the Model C inference path, and makes
the dataset registry's internal split genuinely temporal. 54 tests passing.

## Highlights

### ✨ Spectral POD (SPOD)
New `icarus.decomposition.spod.SPOD` — Welch-method Spectral POD that produces
modes **coherent at a single frequency**, so boiling structures can be
separated by timescale (nucleation / departure / microlayer) instead of being
blended into energy-ranked POD modes. Includes `examples/spod_analysis.py`, a
diagnostic runner that writes an energy-vs-frequency spectrum and leading-mode
spatial maps, with a low-memory loader that reads only the heater z-layer off
disk (~200 MB instead of ~2 GB for the 4-D temperature files).

### 🐛 Fixes
- **`Pipeline._predict_modal` (Model C inference)** — `predict()` on new data
  was double-counting `phi_i` and reshaping time-major rows as pixel-major,
  returning a scrambled field while `evaluate()` stayed correct. Now sums modal
  contributions and reshapes time-major. The deployable predictor works.
- **Extractor ordering** — features are stored time-major so the registry's
  internal train/test split is a real temporal split (was a spatial pixel
  split, which leaked).
- **`Pipeline.fit()`** respects a manually injected `HeatFluxNet`.

### 📚 Docs
- `docs/literature_survey.md` — heat-flux partitioning and probabilistic/UQ
  prediction, with sources.
- `docs/code_walkthrough.md` — file-by-file walkthrough of the Model C path and
  full pipeline.

## ⚠️ Upgrade note
`features.h5` files extracted with 0.1.0 are incompatible with the corrected
time-major ordering — re-run `FeatureExtractor.process(..., force=True)`.

**Full changelog:** see `CHANGELOG.md`.
