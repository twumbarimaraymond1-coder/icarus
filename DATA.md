# Icarus Dataset Registry

Icarus improves with every new dataset. This page explains what data we
are looking for, how to contribute, and what contributors receive in return.

---

## Why contribute?

Every dataset contributed to Icarus:

- Is used to validate and extend the core POD/DMD/ANN pipeline across new
  fluids, surfaces, and operating conditions
- Is credited permanently in this registry and in any publications that use it
- Improves the pre-trained models available to every Icarus user
- Strengthens the case for a community benchmark dataset for IR-based heat
  flux prediction — a resource the field currently lacks

If your dataset is substantial (>500k pixel-time samples), we will invite you
to co-author the Icarus benchmark paper when it is submitted.

---

## What we are looking for

We are primarily interested in **infrared thermography datasets from boiling
and two-phase heat transfer experiments** where both surface temperature and
heat flux fields are available. Datasets covering the following are especially
valuable:

| Priority | Description |
|----------|-------------|
| High | Flow boiling — different fluids (refrigerants, dielectric fluids, FC-72) |
| High | Pool boiling — different surface finishes and enhanced surfaces |
| High | Approach-to-CHF and CHF conditions |
| Medium | Immersion cooling of electronics substrates |
| Medium | Condensation and evaporation on thin-film heaters |
| Lower | Single-phase forced convection (useful for baseline comparisons) |

We are **not** limited to boiling — any transient IR thermography dataset
where a heat flux ground truth exists is worth discussing.

---

## Data format

### Preferred format

NumPy `.npz` or MATLAB `.mat` (v5 or v7.3/HDF5) files containing:

| Variable | Shape | Description |
|----------|-------|-------------|
| `T` | `[ny, nx, nt]` or `[ny, nx, nz, nt]` | Surface temperature field (K or °C) |
| `q` | `[ny, nx, nt]` | Heat flux field (W/m²) |
| `dt` | scalar | Timestep in seconds |

HDF5 (`.h5`) is also accepted. If your data is in a different format, open
an issue and we will add a loader for it.

### Minimum requirements

- Spatial resolution sufficient to resolve individual nucleation sites
  (roughly < 0.5 mm per pixel)
- At least 500 timesteps
- Consistent units documented in a README alongside the data
- No personal or institutional identifiers in the data files themselves

### Metadata to include

Please provide a short `metadata.json` or `README.txt` alongside your data
covering:

```json
{
  "fluid": "water / FC-72 / R134a / ...",
  "surface": "plain copper / micropillar / sintered / ...",
  "setup": "pool boiling / flow boiling / immersion cooling / ...",
  "mean_heat_flux_W_m2": 300000,
  "pressure_bar": 1.013,
  "pixel_size_mm": [0.24, 0.22],
  "frame_rate_Hz": 4000,
  "n_frames": 4000,
  "published_doi": "10.xxxx/xxxxx or null",
  "embargo_months": 0
}
```

---

## Embargo policy

If your dataset accompanies an unpublished paper, we offer a **6-month
embargo** by default (extendable to 12 months on request). During this
period your data is used only for internal model validation and is not
shared publicly. After the embargo expires, data enters the community
pool under the CC BY 4.0 licence unless you request otherwise.

---

## How to contribute

**Option A — Direct upload (recommended for datasets < 5 GB)**

Open a GitHub issue with the title `[Dataset] <fluid> <setup> <your institution>`.
We will share a secure upload link.

**Option B — Point us to your repository**

If your data is already on Zenodo, Figshare, Mendeley Data, or a
university repository, open an issue with the DOI and we will handle
the ingestion ourselves.

**Option C — Federated (data stays on your servers)**

If your institution has data governance restrictions that prevent upload,
we support federated validation — your data never leaves your servers and
only model gradients are shared. Open an issue labelled `[Federated]` to
discuss.

---

## What contributors receive

| Contribution size | Recognition |
|-------------------|-------------|
| Any dataset | Named credit in this registry and in model cards |
| > 500k samples | Invitation to co-author the Icarus benchmark paper |
| > 5M samples | Named acknowledgement in all derivative publications |
| Multiple datasets | Offer of an advisory role in the project |

---

## Current dataset registry

| ID | Fluid | Setup | Source | Samples | Status |
|----|-------|-------|--------|---------|--------|
| D001 | Water | Flow boiling (Loughborough) | Private — project dataset | 17.4M | Active (training) |

*We are actively seeking D002 and beyond. If you have data, please get in touch.*

---

## Contact

Open a [GitHub issue](https://github.com/twumbarimaraymond1-coder/icarus/issues)
or email via the address in the repository profile.

We respond to all dataset enquiries within 5 working days.
