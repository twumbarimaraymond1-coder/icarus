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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

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

    def evaluate(self, verbose: bool = True) -> dict:
        """Evaluate train and test performance.

        Returns
        -------
        dict with per-dataset and combined metrics.
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

        if verbose:
            print(f"\nDatasets: {self._dataset_ids_used}")
            print(m_train)
            print(m_test)

        return {"train": m_train, "test": m_test}

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

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Trainer has not been fitted. Call fit() first."
            )
