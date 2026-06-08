"""
icarus.registry.trainer
========================
Multi-dataset training for the POD modal prediction strategy.

Loads processed HDF5 feature files from multiple datasets, combines
them into a unified training pool, and trains a single HeatFluxNet
that generalises across fluids and surfaces.

Why this works
--------------
Raw temperature values are not comparable across datasets — water at
427 K and FC-72 at 320 K have nothing in common at the pixel level.
But their POD modal contributions (the rank-1 spatial-temporal
structures) capture the *dominant patterns of deviation from the mean*
in a normalised form. Training on these modal features allows the
network to learn relationships that transfer across fluids.

The key assumption (supported by the r ≥ 0.96 correlation found in
the Loughborough dataset) is that the coupling between temperature
and heat flux dominant modes is a general physical feature of boiling,
not a dataset-specific artefact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import h5py

from icarus.models.neural import HeatFluxNet
from icarus.metrics.evaluation import evaluate
from icarus.registry.dataset import DatasetRegistry


class MultiDatasetTrainer:
    """Train a heat flux prediction model across multiple datasets.

    Parameters
    ----------
    registry : DatasetRegistry
    n_pod_modes : int
        Must match the n_pod_modes used during feature extraction.
    n_training_samples : int or None
        Total samples to use for training (subsampled equally across
        datasets). None = use all available samples.
    optimise_hyperparams : bool
    n_trials : int
    random_state : int

    Examples
    --------
    >>> from icarus.registry.dataset import DatasetRegistry
    >>> from icarus.registry.trainer import MultiDatasetTrainer
    >>>
    >>> reg = DatasetRegistry("~/.icarus/datasets")
    >>> trainer = MultiDatasetTrainer(reg, n_pod_modes=5)
    >>>
    >>> # Train on all validated datasets
    >>> model = trainer.fit(dataset_ids=["D001", "D002", "D003"])
    >>> trainer.evaluate()
    >>> trainer.save_model("models/multi_dataset_v1.pkl")
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        n_pod_modes: int = 5,
        n_training_samples: Optional[int] = 1_000_000,
        optimise_hyperparams: bool = True,
        n_trials: int = 30,
        random_state: int = 42,
    ):
        self.registry = registry
        self.n_pod_modes = n_pod_modes
        self.n_training_samples = n_training_samples
        self.optimise_hyperparams = optimise_hyperparams
        self.n_trials = n_trials
        self.random_state = random_state

        self.model_: Optional[HeatFluxNet] = None
        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._dataset_ids_used: list[str] = []
        self._test_dataset_id: Optional[str] = None
        self._q_abs_test: Optional[np.ndarray] = None
        self._fitted = False

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        dataset_ids: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> "MultiDatasetTrainer":
        """Load features and train the model.

        Parameters
        ----------
        dataset_ids : list[str], optional
            Specific dataset IDs to train on. If None, uses all datasets
            with status ``"features_extracted"`` or ``"validated"``.
        verbose : bool

        Returns
        -------
        self
        """
        # Resolve dataset list
        if dataset_ids is None:
            entries = self.registry.list_datasets()
            dataset_ids = [
                e.dataset_id for e in entries
                if e.status in ("features_extracted", "validated")
                and e.processed_path is not None
            ]

        if not dataset_ids:
            raise ValueError(
                "No processed datasets found. "
                "Run FeatureExtractor.process() on your datasets first."
            )

        if verbose:
            print(f"[trainer] Loading features from {len(dataset_ids)} "
                  f"dataset(s): {dataset_ids}")

        # ── Load and combine features ─────────────────────────────────────
        X_train_parts, X_test_parts = [], []
        y_train_parts, y_test_parts = [], []

        for ds_id in dataset_ids:
            entry = self.registry.get(ds_id)
            if not entry.processed_path:
                print(f"  Skipping {ds_id} — not yet processed.")
                continue

            X_tr, X_te, y_tr, y_te = self._load_features(
                entry.processed_path, verbose=verbose
            )
            X_train_parts.append(X_tr)
            X_test_parts.append(X_te)
            y_train_parts.append(y_tr)
            y_test_parts.append(y_te)
            self._dataset_ids_used.append(ds_id)

        if not X_train_parts:
            raise ValueError("No features could be loaded.")

        X_train = np.concatenate(X_train_parts, axis=0)
        X_test = np.concatenate(X_test_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        y_test = np.concatenate(y_test_parts, axis=0)

        if verbose:
            print(f"[trainer] Combined training set: "
                  f"{len(X_train):,} samples from "
                  f"{len(dataset_ids)} dataset(s)")
            print(f"[trainer] Combined test set:     "
                  f"{len(X_test):,} samples")

        # Subsample if requested
        if self.n_training_samples and len(X_train) > self.n_training_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_train), self.n_training_samples,
                             replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            if verbose:
                print(f"[trainer] Subsampled to "
                      f"{self.n_training_samples:,} training samples")

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        # ── Train model ───────────────────────────────────────────────────
        # Use pre-set model if provided, otherwise create a new one
        if self.model_ is None:
            self.model_ = HeatFluxNet(
                strategy="modal",
                random_state=self.random_state,
            )

        if self.optimise_hyperparams and self.model_.hidden_layer_sizes is None:
            if verbose:
                print(f"[trainer] Bayesian optimisation "
                      f"({self.n_trials} trials)...")
            self.model_.optimise(
                X_train, y_train,
                n_trials=self.n_trials,
                verbose=verbose,
            )

        if verbose:
            print("[trainer] Training final model...")
        self.model_.fit(X_train, y_train,
                        n_samples=self.n_training_samples)

        self._fitted = True
        return self

    def cross_dataset_fit(
        self,
        train_ids: list[str],
        test_id: str,
        verbose: bool = True,
    ) -> "MultiDatasetTrainer":
        """Train on two (or more) datasets and evaluate on a held-out third.

        Unlike ``fit()``, which uses each dataset's own internal train/test
        split, this method uses *all* samples from ``train_ids`` for training
        and *all* samples from ``test_id`` for testing. The result is a
        stricter cross-dataset generalisation evaluation: the model never sees
        any data from the test dataset during training. This is the right
        protocol for answering "does the model generalise to an unseen heater
        surface / fluid?" rather than merely to unseen timesteps.

        Parameters
        ----------
        train_ids : list[str]
            Dataset IDs whose complete data is used for training
            (e.g. ``["D001", "D002"]``).
        test_id : str
            Dataset ID held out entirely for evaluation (e.g. ``"D003"``).
        verbose : bool

        Returns
        -------
        self

        Examples
        --------
        >>> trainer = MultiDatasetTrainer(registry, n_pod_modes=5)
        >>> trainer.cross_dataset_fit(train_ids=["D001", "D002"], test_id="D003")
        >>> trainer.evaluate()
        """
        if not train_ids:
            raise ValueError("train_ids must contain at least one dataset ID.")
        if test_id in train_ids:
            raise ValueError(
                f"test_id '{test_id}' must not appear in train_ids — "
                "it would leak test data into training."
            )

        if verbose:
            print("[trainer] Cross-dataset split:")
            print(f"  Train : {train_ids}")
            print(f"  Test  : {test_id}")

        # ── Load ALL samples from training datasets ───────────────────────
        X_train_parts, y_train_parts = [], []
        for ds_id in train_ids:
            entry = self.registry.get(ds_id)
            if not entry.processed_path:
                raise ValueError(
                    f"Dataset '{ds_id}' has not been processed. "
                    "Run FeatureExtractor.process() first."
                )
            X_all, y_all = self._load_all_features(
                entry.processed_path, verbose=verbose
            )
            X_train_parts.append(X_all)
            y_train_parts.append(y_all)
            self._dataset_ids_used.append(ds_id)

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)

        # ── Load ALL samples from the held-out test dataset ───────────────
        test_entry = self.registry.get(test_id)
        if not test_entry.processed_path:
            raise ValueError(
                f"Test dataset '{test_id}' has not been processed. "
                "Run FeatureExtractor.process() first."
            )
        X_test, y_test = self._load_all_features(
            test_entry.processed_path, verbose=verbose
        )
        # Absolute (un-centred) ground-truth heat flux for the held-out set,
        # row-aligned with X_test/y_test. Used for absolute-field R² in
        # evaluate(). None if the features file predates this field.
        self._q_abs_test = self._load_q_abs(test_entry.processed_path)

        if verbose:
            print(f"[trainer] Training samples : {len(X_train):,}")
            print(f"[trainer] Test samples     : {len(X_test):,}  "
                  f"(from {test_id})")

        # Subsample training data if requested
        if self.n_training_samples and len(X_train) > self.n_training_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_train), self.n_training_samples,
                             replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            if verbose:
                print(f"[trainer] Subsampled to "
                      f"{self.n_training_samples:,} training samples")

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._test_dataset_id = test_id

        # ── Train model ───────────────────────────────────────────────────
        if self.model_ is None:
            self.model_ = HeatFluxNet(
                strategy="modal",
                random_state=self.random_state,
            )

        if self.optimise_hyperparams and self.model_.hidden_layer_sizes is None:
            if verbose:
                print(f"[trainer] Bayesian optimisation "
                      f"({self.n_trials} trials)...")
            self.model_.optimise(
                X_train, y_train,
                n_trials=self.n_trials,
                verbose=verbose,
            )

        if verbose:
            print("[trainer] Training final model...")
        self.model_.fit(X_train, y_train, n_samples=self.n_training_samples)

        self._fitted = True
        return self

    def evaluate(self, verbose: bool = True) -> dict:
        """Evaluate train and test performance.

        Two flavours of metric are reported for the test set:

        * **fluctuation** R²/RMSE — computed on the mean-subtracted (centred)
          heat-flux field that the POD modal model actually predicts. This is
          the scientifically meaningful number: it measures how well the
          temperature→heat-flux *modal coupling* transfers. The returned
          ``"test"`` key is this fluctuation metric (also under
          ``"test_fluctuation"``).
        * **absolute** R²/RMSE — computed on the full un-centred field
          (per-pixel mean added back). Always looks more favourable because the
          spatial mean dominates the variance; reported under
          ``"test_absolute"`` for transparency. Only present when the features
          file carries the ``q_abs`` array (re-extract older datasets to get it).

        Returns
        -------
        dict with keys ``"train"``, ``"test"``, ``"test_fluctuation"`` and,
        when available, ``"test_absolute"``.
        """
        self._require_fit()

        y_train_pred = self.model_.predict(self._X_train)
        y_test_pred = self.model_.predict(self._X_test)

        m_train = evaluate(
            self._y_train.sum(axis=1),
            y_train_pred.sum(axis=1),
            split="train",
        )
        m_test = evaluate(
            self._y_test.sum(axis=1),
            y_test_pred.sum(axis=1),
            split="test",
        )

        result = {"train": m_train, "test": m_test, "test_fluctuation": m_test}

        # Absolute-field R²: add the per-pixel mean back to both sides. The
        # offset (q_abs - centred truth) cancels in the residual, so the model
        # is scored on its modal error against the full-field variance — a fair
        # absolute metric that does not penalise POD truncation.
        m_test_abs = None
        if (self._q_abs_test is not None
                and len(self._q_abs_test) == len(self._y_test)):
            true_centred = self._y_test.sum(axis=1)
            offset = self._q_abs_test - true_centred
            pred_abs = y_test_pred.sum(axis=1) + offset
            m_test_abs = evaluate(self._q_abs_test, pred_abs, split="test")
            result["test_absolute"] = m_test_abs

        if verbose:
            print(f"\nTrain datasets: {self._dataset_ids_used}")
            if self._test_dataset_id is not None:
                print(f"Held-out test dataset: {self._test_dataset_id}")
            print(m_train)
            print(f"[test · fluctuation field] {m_test}")
            if m_test_abs is not None:
                print(f"[test · absolute field]    {m_test_abs}")

        return result

    def save_model(self, path: str | Path) -> None:
        """Save the trained model to disk.

        Parameters
        ----------
        path : str or Path
            Output path. Extension ``.pkl`` recommended.
        """
        self._require_fit()
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model_,
                "dataset_ids": self._dataset_ids_used,
                "n_pod_modes": self.n_pod_modes,
            }, f)
        print(f"[trainer] Model saved to {path}")

    @staticmethod
    def load_model(path: str | Path) -> dict:
        """Load a saved model.

        Returns
        -------
        dict with keys ``"model"``, ``"dataset_ids"``, ``"n_pod_modes"``.
        """
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_features(
        self,
        features_path: str,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load train/test splits from a processed HDF5 features file."""
        with h5py.File(features_path, "r") as f:
            X = f["T_contribs"][:]     # [n_samples, n_modes]
            y = f["q_contribs"][:]     # [n_samples, n_modes]
            split = f["split"][:]      # 0=train, 1=test
            ds_id = f.attrs.get("dataset_id", "?")
            n_samples = f.attrs.get("n_samples", len(X))

        train_mask = split == 0
        test_mask = split == 1

        if verbose:
            print(f"  {ds_id}: {train_mask.sum():,} train  "
                  f"{test_mask.sum():,} test  "
                  f"({n_samples:,} total)")

        return (
            X[train_mask], X[test_mask],
            y[train_mask], y[test_mask],
        )

    def _load_all_features(
        self,
        features_path: str,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load *all* samples from a processed HDF5 file, ignoring the
        internal train/test split mask.

        Used by ``cross_dataset_fit`` so that entire datasets contribute
        fully to either training or testing, never a sub-split of each.

        Returns
        -------
        X : np.ndarray, shape [n_samples, n_modes]
        y : np.ndarray, shape [n_samples, n_modes]
        """
        with h5py.File(features_path, "r") as f:
            X = f["T_contribs"][:]     # [n_samples, n_modes]
            y = f["q_contribs"][:]     # [n_samples, n_modes]
            ds_id = f.attrs.get("dataset_id", "?")
            n_samples = len(X)

        if verbose:
            print(f"  {ds_id}: {n_samples:,} samples loaded (all)")

        return X, y

    def _load_q_abs(
        self,
        features_path: str,
    ) -> Optional[np.ndarray]:
        """Load the absolute (un-centred) heat-flux target if present.

        Stored pixel-major by the extractor so it is row-aligned with
        ``q_contribs``/``T_contribs``. Returns None for older feature files
        that do not contain ``q_abs`` (absolute R² is then skipped gracefully).
        """
        with h5py.File(features_path, "r") as f:
            if "q_abs" not in f:
                return None
            return f["q_abs"][:]

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Trainer has not been fitted. Call fit() first."
            )
