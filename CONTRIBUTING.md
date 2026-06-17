# Contributing to Icarus

Thank you for your interest in contributing. Icarus is an early-stage
open-source library and contributions of all kinds are welcome — bug fixes,
new features, additional datasets, documentation improvements, and
real-world validation results.

---

## Getting started

```bash
git clone https://github.com/twumbarimaraymond1-coder/icarus
cd icarus
pip install -e ".[dev]"
```

Run the test suite to make sure everything works:

```bash
pytest tests/ -v
```

---

## What we especially need

**Additional experimental datasets.** The library was developed on a single
flow boiling experiment. Contributions of IR thermography data from different
fluids, surfaces, and operating conditions — even small datasets — are
enormously valuable for training more general models. If you have data you
are willing to share, please open an issue to discuss the contribution format.

**New fluid/surface validations.** If you run Icarus on your own experimental
data and get results (good or bad), please share them as a GitHub issue or
discussion. Real-world validation results are the most useful contribution
at this stage.

**DMD improvements.** The current DMD implementation handles short-horizon
forecasting reasonably but accuracy degrades over longer horizons. Improved
variants (optimised DMD, DMD with control, Hankel-DMD) would be welcome
additions.

**Pre-trained model contributions.** If you train a model on a new dataset
and are willing to share the weights and a description of the training
conditions, we can add it to a community model registry.

---

## How to contribute code

1. Fork the repository and create a branch from `main`.
2. Make your changes, following the code style below.
3. Add tests for any new functionality in `tests/`.
4. Run `pytest tests/ -v` and ensure all tests pass.
5. Open a pull request with a clear description of what you changed and why.

---

## Design principle: features must be user-friendly

Icarus is used by researchers, not software engineers. **Every new feature
must expose a high-level, few-lines-of-code API that mirrors the Quickstart**,
not just a low-level primitive. The model is `tf.Pipeline` and `tf.SPOD`:

- The headline usage of a feature should read like a recipe — a handful of
  clear calls — with all machinery (loading, preprocessing, bookkeeping,
  plotting) hidden inside the package, never re-implemented in the user's
  script.
- **Design the ~5-line usage first**, then build the implementation behind it.
- Provide a one-call convenience that wraps the low-level steps (e.g.
  `SPOD.fit_field` wraps crop/centre/flatten + `fit`), while keeping the
  low-level primitive available for power users.
- Export the main entry points at the top level (`icarus.X`) so they are
  discoverable as `tf.X`.
- Ship plotting/IO as methods or helpers (lazy-import optional deps like
  matplotlib), so users never write boilerplate.
- Document the feature with a Quickstart-style snippet in the README, add a
  runnable `examples/` script, and **test the high-level path**, not only the
  internals.

If a feature can only be used by writing a long script, it is not finished.

---

## Code style

- Follow PEP 8. We use `ruff` for linting (`ruff check .`).
- All public functions and classes must have NumPy-style docstrings.
- Type hints on all function signatures.
- Keep modules focused — if you are adding a substantially new capability
  (e.g. a new decomposition method), add a new file rather than extending
  an existing one.

---

## Reporting bugs

Open a GitHub issue with:
- Your Python version and OS
- The exact error message and traceback
- A minimal reproducible example (synthetic data is fine)

---

## Data sharing and privacy

If you are contributing experimental data, please ensure:
- You have the right to share it (check with your institution / supervisor)
- Any identifying metadata has been removed
- You are happy for the data to be used to train community models under
  the CC BY 4.0 licence

---

## Code of conduct

Be kind. This is a research-focused project. Disagreements about methodology
are welcome and expected — personal criticism is not.
