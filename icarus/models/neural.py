"""
icarus.models.neural
========================
Multilayer perceptron for heat flux prediction with Bayesian hyperparameter
optimisation via Optuna.

Three model strategies from the paper are supported:

- ``"raw"``     – Model A: raw temperature input
- ``"gradient"``– Model B: temperature + spatial/temporal gradients
- ``"modal"``   – Model C: POD modal contributions (best performance)
"""

from __future__ import annotations

from typing import Optional, Literal, Union
import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin


Strategy = Literal["raw", "gradient", "modal"]

SEARCH_SPACES: dict[str, dict] = {
    "small": {
        "n_layers_min": 1, "n_layers_max": 2,
        "n_units_min": 16, "n_units_max": 128, "n_units_step": 16,
        "activations": ["relu", "tanh"],
        "alpha_min": 1e-5, "alpha_max": 1e-2,
        "lr_min": 1e-4, "lr_max": 1e-2,
    },
    "medium": {
        "n_layers_min": 1, "n_layers_max": 3,
        "n_units_min": 16, "n_units_max": 256, "n_units_step": 16,
        "activations": ["relu", "tanh"],
        "alpha_min": 1e-5, "alpha_max": 1e-1,
        "lr_min": 1e-4, "lr_max": 1e-2,
    },
    "large": {
        "n_layers_min": 2, "n_layers_max": 5,
        "n_units_min": 64, "n_units_max": 512, "n_units_step": 32,
        "activations": ["relu", "tanh"],
        "alpha_min": 1e-7, "alpha_max": 1e-1,
        "lr_min": 1e-5, "lr_max": 5e-2,
    },
}
_DEFAULT_SEARCH_SPACE = "medium"


class HeatFluxNet(BaseEstimator, RegressorMixin):
    """MLP-based heat flux predictor with optional Bayesian tuning.

    Parameters
    ----------
    strategy : str
        One of ``"raw"``, ``"gradient"``, ``"modal"``. Used for
        documentation and pipeline labelling only — feature construction
        is handled by :mod:`icarus.features.engineer`.
    hidden_layer_sizes : tuple[int, ...], optional
        ANN architecture. If None, must call :meth:`optimise` first or
        pass sizes explicitly.
    activation : str
        Activation function: ``"relu"`` or ``"tanh"``.
    alpha : float
        L2 regularisation strength.
    learning_rate_init : float
        Initial learning rate for the Adam solver.
    max_iter : int
        Maximum training iterations.
    random_state : int
        Reproducibility seed.

    Examples
    --------
    >>> from icarus.models.neural import HeatFluxNet
    >>> model = HeatFluxNet(strategy="modal")
    >>> model.optimise(X_train, y_train, n_trials=30)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        strategy: Strategy = "modal",
        hidden_layer_sizes: Optional[tuple] = None,
        activation: str = "tanh",
        alpha: float = 1e-3,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._mlp: Optional[MLPRegressor] = None
        self._best_params: Optional[dict] = None
        self._fitted = False

    # ── public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _subsample_indices(
        n_total: int,
        n: int,
        random_state: int,
        validation_strategy: str,
    ) -> np.ndarray:
        """Pick ``n`` row indices out of ``n_total`` for trial evaluation.

        For ``validation_strategy == "temporal"`` the selected indices are
        returned in ascending (chronological) order so the downstream 80/20
        head/tail split is a genuine past→future split. For any other strategy
        the indices are returned in random order (the caller applies an i.i.d.
        sklearn split, so order is irrelevant there).
        """
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_total, size=n, replace=False)
        if validation_strategy == "temporal":
            idx = np.sort(idx)
        return idx

    def optimise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 30,
        n_opt_samples: int = 200_000,
        search_space: Optional[Union[str, dict]] = None,
        validation_strategy: str = "random",
        verbose: bool = True,
    ) -> dict:
        """Run Bayesian hyperparameter optimisation with Optuna.

        Uses a random subsample of the data for fast trial evaluation.
        Best parameters are stored and used automatically by :meth:`fit`.

        Parameters
        ----------
        X : np.ndarray, shape [n_samples, n_features]
        y : np.ndarray, shape [n_samples] or [n_samples, n_targets]
        n_trials : int
            Number of Optuna trials.
        n_opt_samples : int
            Subsample size for each trial (speeds up optimisation).
        search_space : dict, optional
            Custom search space. Keys: ``n_layers``, ``n_units``,
            ``activation``, ``alpha``, ``lr_init``.
        verbose : bool
            Print best parameters when complete.

        Returns
        -------
        dict : Best hyperparameters found.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna is required for hyperparameter optimisation.")

        from sklearn.model_selection import train_test_split as sk_split
        from sklearn.metrics import r2_score

        if search_space is None:
            sp = SEARCH_SPACES[_DEFAULT_SEARCH_SPACE]
        elif isinstance(search_space, str):
            if search_space not in SEARCH_SPACES:
                raise ValueError(
                    f"Unknown search space preset '{search_space}'. "
                    f"Choose from: {list(SEARCH_SPACES.keys())} or pass a dict."
                )
            sp = SEARCH_SPACES[search_space]
        else:
            sp = search_space

        # Subsample for speed
        n = min(n_opt_samples, len(X))
        idx = self._subsample_indices(
            len(X), n, self.random_state, validation_strategy
        )
        X_opt, y_opt = X[idx], y[idx]

        if validation_strategy == "temporal":
            split_idx = int(len(X_opt) * 0.8)
            X_tr, X_val = X_opt[:split_idx], X_opt[split_idx:]
            y_tr, y_val = y_opt[:split_idx], y_opt[split_idx:]
        else:
            X_tr, X_val, y_tr, y_val = sk_split(
                X_opt, y_opt, test_size=0.2, random_state=self.random_state
            )

        # Scale inside objective to avoid leakage
        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1) if y_tr.ndim == 1 else y_tr)

        X_tr_s = sx.transform(X_tr)
        X_val_s = sx.transform(X_val)
        y_tr_s = sy.transform(y_tr.reshape(-1, 1) if y_tr.ndim == 1 else y_tr)
        y_val_raw = y_val

        def objective(trial: "optuna.Trial") -> float:
            n_layers = trial.suggest_int(
                "n_layers", sp.get("n_layers_min", 1), sp.get("n_layers_max", 3)
            )
            n_units = trial.suggest_int(
                "n_units",
                sp.get("n_units_min", 16),
                sp.get("n_units_max", 256),
                step=16,
            )
            activation = trial.suggest_categorical(
                "activation", sp.get("activations", ["relu", "tanh"])
            )
            alpha = trial.suggest_float(
                "alpha",
                sp.get("alpha_min", 1e-5),
                sp.get("alpha_max", 1e-1),
                log=True,
            )
            lr_init = trial.suggest_float(
                "lr_init",
                sp.get("lr_min", 1e-4),
                sp.get("lr_max", 1e-2),
                log=True,
            )

            mlp = MLPRegressor(
                hidden_layer_sizes=tuple([n_units] * n_layers),
                activation=activation,
                solver="adam",
                alpha=alpha,
                learning_rate_init=lr_init,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp.fit(X_tr_s, y_tr_s.ravel() if y_tr_s.ndim == 1 else y_tr_s)

            y_pred = sy.inverse_transform(
                mlp.predict(X_val_s).reshape(-1, 1)
                if y_tr_s.ndim == 1
                else mlp.predict(X_val_s)
            )
            return float(r2_score(y_val_raw, y_pred, multioutput="uniform_average"))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        best = study.best_params
        self._best_params = best

        # Apply best params
        self.hidden_layer_sizes = tuple([best["n_units"]] * best["n_layers"])
        self.activation = best["activation"]
        self.alpha = best["alpha"]
        self.learning_rate_init = best["lr_init"]

        if verbose:
            print("\n=== Best hyperparameters ===")
            for k, v in best.items():
                print(f"  {k}: {v}")
            print(f"  validation R²: {study.best_value:.4f}")

        return best

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: Optional[int] = 1_000_000,
    ) -> "HeatFluxNet":
        """Train the model.

        Parameters
        ----------
        X : np.ndarray, shape [n_samples, n_features]
        y : np.ndarray, shape [n_samples] or [n_samples, n_targets]
        n_samples : int or None
            Subsample this many rows for training (speeds up fitting on
            large datasets). Set to None to use all data.

        Returns
        -------
        self
        """
        if self.hidden_layer_sizes is None:
            raise RuntimeError(
                "hidden_layer_sizes is not set. "
                "Either pass it to __init__ or call optimise() first."
            )

        if n_samples is not None and len(X) > n_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), size=n_samples, replace=False)
            X, y = X[idx], y[idx]

        # Fit scalers
        self._scaler_X.fit(X)
        y2 = y.reshape(-1, 1) if y.ndim == 1 else y
        self._scaler_y.fit(y2)

        X_s = self._scaler_X.transform(X)
        y_s = self._scaler_y.transform(y2)

        self._mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._mlp.fit(X_s, y_s.ravel() if y_s.ndim == 1 else y_s)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict heat flux (or modal contributions for Model C).

        Parameters
        ----------
        X : np.ndarray, shape [n_samples, n_features]

        Returns
        -------
        np.ndarray, shape [n_samples] or [n_samples, n_targets]
        """
        self._require_fit()
        X_s = self._scaler_X.transform(X)
        y_s = self._mlp.predict(X_s)
        y2 = y_s.reshape(-1, 1) if y_s.ndim == 1 else y_s
        y_out = self._scaler_y.inverse_transform(y2)
        return y_out.ravel() if y_out.shape[1] == 1 else y_out

    # ── private helpers ───────────────────────────────────────────────────────

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
