"""
icarus.pipeline.runner
==========================
End-to-end Pipeline that orchestrates preprocessing, decomposition,
feature construction, model training, and evaluation in a single object.

The Pipeline defaults to the Model C (POD modal contributions) strategy
which gave the best results in the paper (R² = 0.729 on the test set).
"""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np

from icarus.data.preprocessor import Preprocessor, PreprocessorConfig
from icarus.decomposition.pod import POD
from icarus.features.engineer import (
    build_raw_features,
    build_gradient_features,
    build_modal_features,
    flatten_target,
    train_test_split_temporal,
)
from icarus.models.neural import HeatFluxNet
from icarus.metrics.evaluation import evaluate, evaluate_timeresolved, Metrics


Strategy = Literal["raw", "gradient", "modal"]


class Pipeline:
    """Full heat flux prediction pipeline.

    Parameters
    ----------
    strategy : str
        Feature strategy — ``"raw"`` (Model A), ``"gradient"`` (Model B),
        or ``"modal"`` (Model C, default).
    n_pod_modes : int
        Number of POD modes used for feature construction (Model C only).
    spatial_crop : int
        Pixels to crop from each spatial boundary during preprocessing.
    trim_frames : int
        Trailing frames to discard during preprocessing.
    train_fraction : float
        Fraction of timesteps used for training.
    n_training_samples : int or None
        Subsample for ANN training. None = use all.
    optimise_hyperparams : bool
        Whether to run Bayesian optimisation before training.
    n_trials : int
        Number of Optuna trials for hyperparameter optimisation.
    random_state : int

    Examples
    --------
    >>> import icarus as tf
    >>>
    >>> data = tf.data.loader.load("experiment.mat")
    >>>
    >>> pipe = tf.Pipeline(strategy="modal", n_pod_modes=5)
    >>> pipe.fit(data)
    >>> metrics = pipe.evaluate()
    >>> print(metrics["test"])
    [test]  R² = 0.7293  RMSE = 25,959 W/m²  MAE = 20,656 W/m²
    """

    def __init__(
        self,
        strategy: Strategy = "modal",
        n_pod_modes: int = 5,
        spatial_crop: int = 5,
        trim_frames: int = 0,
        train_fraction: float = 0.7,
        n_training_samples: Optional[int] = 1_000_000,
        optimise_hyperparams: bool = True,
        n_trials: int = 30,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.n_pod_modes = n_pod_modes
        self.spatial_crop = spatial_crop
        self.trim_frames = trim_frames
        self.train_fraction = train_fraction
        self.n_training_samples = n_training_samples
        self.optimise_hyperparams = optimise_hyperparams
        self.n_trials = n_trials
        self.random_state = random_state

        # Components — populated during fit()
        self.preprocessor_: Optional[Preprocessor] = None
        self.pod_T_: Optional[POD] = None
        self.pod_q_: Optional[POD] = None
        self.model_: Optional[HeatFluxNet] = None

        # Data splits stored for evaluation
        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._processed: Optional[dict] = None
        self._fitted = False

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, data: dict[str, np.ndarray], verbose: bool = True) -> "Pipeline":
        """Fit the full pipeline on a thermography dataset.

        Parameters
        ----------
        data : dict
            Output of :func:`icarus.data.loader.load`.
        verbose : bool

        Returns
        -------
        self
        """
        if verbose:
            print(f"[icarus] Strategy: {self.strategy}")

        # 1. Preprocessing
        if verbose:
            print("[icarus] Preprocessing...")
        cfg = PreprocessorConfig(
            spatial_crop=self.spatial_crop,
            trim_frames=self.trim_frames,
        )
        self.preprocessor_ = Preprocessor(cfg)
        processed = self.preprocessor_.fit_transform(data)
        self._processed = processed

        T, q = processed["T"], processed["q"]
        T_c, q_c = processed["T_c"], processed["q_c"]
        ny, nx, nt = T.shape

        # 2. POD decomposition
        if verbose:
            print(f"[icarus] Running POD ({self.n_pod_modes} modes)...")
        X_c_T = Preprocessor.to_matrix(T_c)
        X_c_q = Preprocessor.to_matrix(q_c)

        self.pod_T_ = POD(n_modes=self.n_pod_modes).fit(X_c_T)
        self.pod_q_ = POD(n_modes=self.n_pod_modes).fit(X_c_q)

        if verbose:
            cum_T = self.pod_T_.cumulative_energy_[self.n_pod_modes - 1]
            cum_q = self.pod_q_.cumulative_energy_[self.n_pod_modes - 1]
            print(f"  T  energy (first {self.n_pod_modes} modes): {cum_T:.1%}")
            print(f"  q  energy (first {self.n_pod_modes} modes): {cum_q:.1%}")

        # 3. Feature and target construction
        if verbose:
            print("[icarus] Building features...")
        X, y = self._build_features(T, q, T_c, q_c, X_c_T, X_c_q, ny, nx, nt)

        # 4. Temporal train/test split
        X_train, X_test, y_train, y_test = train_test_split_temporal(
            X, y,
            train_fraction=self.train_fraction,
            n_pix=ny * nx,
            nt=nt,
        )
        self._X_train, self._X_test = X_train, X_test
        self._y_train, self._y_test = y_train, y_test

        # 5. Model
        self.model_ = HeatFluxNet(strategy=self.strategy, random_state=self.random_state)

        if self.optimise_hyperparams:
            if verbose:
                print(f"[icarus] Bayesian optimisation ({self.n_trials} trials)...")
            self.model_.optimise(
                X_train, y_train,
                n_trials=self.n_trials,
                verbose=verbose,
            )

        if verbose:
            print("[icarus] Training model...")
        self.model_.fit(X_train, y_train, n_samples=self.n_training_samples)

        self._fitted = True
        if verbose:
            print("[icarus] Done.")
        return self

    def predict(self, T_new: np.ndarray) -> np.ndarray:
        """Predict heat flux from a new temperature field.

        Parameters
        ----------
        T_new : np.ndarray, shape [ny, nx, nt_new]
            New temperature measurements (not yet mean-subtracted).

        Returns
        -------
        np.ndarray, shape [ny, nx, nt_new]
            Predicted heat flux field.
        """
        self._require_fit()
        ny, nx, nt = T_new.shape
        T_c_new = T_new - self.preprocessor_.T_mean[:, :, np.newaxis]

        if self.strategy == "modal":
            return self._predict_modal(T_c_new, ny, nx, nt)
        elif self.strategy == "gradient":
            X = build_gradient_features(T_new, dt=self._processed["dt"])
            q_flat = self.model_.predict(X)
            return q_flat.reshape(ny, nx, nt)
        else:  # raw
            X = build_raw_features(T_new)
            q_flat = self.model_.predict(X)
            return q_flat.reshape(ny, nx, nt)

    def evaluate(self) -> dict[str, Metrics]:
        """Compute train and test set metrics.

        Returns
        -------
        dict with keys ``"train"`` and ``"test"``.
        """
        self._require_fit()

        y_train_pred = self.model_.predict(self._X_train)
        y_test_pred = self.model_.predict(self._X_test)

        # For modal strategy, reconstruct heat flux from modal predictions
        if self.strategy == "modal":
            y_train_pred = self._reconstruct_from_modal_preds(
                y_train_pred, split="train"
            )
            y_test_pred = self._reconstruct_from_modal_preds(
                y_test_pred, split="test"
            )

        m_train = evaluate(self._y_train_scalar, y_train_pred, split="train")
        m_test = evaluate(self._y_test_scalar, y_test_pred, split="test")

        print(m_train)
        print(m_test)
        return {"train": m_train, "test": m_test}

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_features(self, T, q, T_c, q_c, X_c_T, X_c_q, ny, nx, nt):
        if self.strategy == "raw":
            X = build_raw_features(T)
            y = flatten_target(q)
            self._y_train_scalar = None
            self._y_test_scalar = None
            return X, y

        elif self.strategy == "gradient":
            X = build_gradient_features(T, dt=self._processed["dt"])
            y = flatten_target(q)
            self._y_train_scalar = None
            self._y_test_scalar = None
            return X, y

        else:  # modal
            T_contribs = self.pod_T_.modal_contributions(X_c_T, n_modes=self.n_pod_modes)
            q_contribs = self.pod_q_.modal_contributions(X_c_q, n_modes=self.n_pod_modes)
            X, y = build_modal_features(T_contribs, q_contribs)
            # Store scalar targets for evaluation
            nt_train = int(nt * self.train_fraction)
            q_flat = flatten_target(q)
            self._y_train_scalar = q_flat[: nt_train * ny * nx]
            self._y_test_scalar = q_flat[nt_train * ny * nx :]
            return X, y

    def _predict_modal(self, T_c_new, ny, nx, nt):
        X_c_new = Preprocessor.to_matrix(T_c_new)
        T_contribs_new = self.pod_T_.modal_contributions(X_c_new, n_modes=self.n_pod_modes)
        X_new, _ = build_modal_features(T_contribs_new)
        q_modal_pred = self.model_.predict(X_new)  # [n_pix*nt, n_modes]

        # Reconstruct heat flux field from predicted modal contributions
        q_modal_pred = q_modal_pred.reshape(ny * nx, nt, self.n_pod_modes)
        q_c_pred = np.zeros((ny * nx, nt))
        for i in range(self.n_pod_modes):
            phi_i = self.pod_q_.modes_[:, i]
            q_c_pred += q_modal_pred[:, :, i] * phi_i[:, np.newaxis]  # approximate

        q_pred = q_c_pred + self.preprocessor_.q_mean.reshape(-1)[:, np.newaxis]
        return q_pred.reshape(ny, nx, nt)

    def _reconstruct_from_modal_preds(self, y_modal, split):
        """Reconstruct scalar heat flux from predicted modal contributions.
        
        y_modal shape: [n_pix*nt, n_modes] — predicted modal contributions
        Returns: [n_pix*nt] — reconstructed heat flux (mean added back)
        """
        T = self._processed["T"]
        ny, nx, nt = T.shape
        nt_train = int(nt * self.train_fraction)
        nt_split = nt_train if split == "train" else (nt - nt_train)

        # Sum modal contributions across modes → q_c field
        q_c_pred = y_modal.sum(axis=1)                          # [n_pix*nt_split]
        q_c_pred = q_c_pred.reshape(ny * nx, nt_split)          # [n_pix, nt_split]

        # Add mean back
        q_mean_vec = self.preprocessor_.q_mean.reshape(-1)
        q_pred = (q_c_pred + q_mean_vec[:, None]).reshape(-1)
        return q_pred

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Pipeline has not been fitted. Call fit() first.")
