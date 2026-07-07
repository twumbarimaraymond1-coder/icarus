"""
icarus.models.probabilistic
===============================
Deep-ensemble heat-flux prediction — a mean prediction plus uncertainty.

Boiling is intrinsically stochastic (random nucleation), so a single point
estimate overstates what any model can know. A **deep ensemble** trains several
independently-seeded networks; where they agree the prediction is confident
(narrow band), where they disagree it is uncertain (wide band). The spread is
the uncertainty estimate — the robust default in the critical-heat-flux
uncertainty literature, and what a CHF-margin warning needs.

For ``strategy="modal"`` each member predicts POD modal contributions; these
are summed across modes (they already include the spatial modes) to give a
scalar heat-flux fluctuation per sample, then the ensemble statistics are taken
over members.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from icarus.models.neural import HeatFluxNet


class ProbabilisticHeatFluxNet:
    """Deep ensemble of :class:`HeatFluxNet` models.

    Parameters
    ----------
    n_members : int
        Number of independently-seeded networks in the ensemble.
    strategy : str
        Passed to each member ("raw", "gradient", "modal").
    random_state : int
        Seeds the per-member seeds (reproducible).
    **net_kwargs
        Forwarded to each :class:`HeatFluxNet` (e.g. ``hidden_layer_sizes``,
        ``max_iter``).

    Examples
    --------
    >>> model = ProbabilisticHeatFluxNet(n_members=5,
    ...                                  hidden_layer_sizes=(64,)).fit(X, y)
    >>> mean = model.predict(X_test)
    >>> mean, lower, upper = model.predict_interval(X_test, coverage=0.9)
    """

    def __init__(
        self,
        n_members: int = 5,
        strategy: str = "modal",
        random_state: int = 42,
        **net_kwargs,
    ):
        if n_members < 2:
            raise ValueError("n_members must be >= 2 for an ensemble.")
        self.n_members = n_members
        self.strategy = strategy
        self.random_state = random_state
        self.net_kwargs = net_kwargs
        self.members_: list[HeatFluxNet] = []
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: Optional[int] = None,
        optimise: bool = False,
        n_trials: int = 20,
    ) -> "ProbabilisticHeatFluxNet":
        """Train all ensemble members.

        Parameters
        ----------
        X, y : np.ndarray
        n_samples : int or None
            Subsample size per member (passed to ``HeatFluxNet.fit``).
        optimise : bool
            Optimise hyperparameters once, then share across members (members
            still differ by random seed). Only used if no architecture is set.
        n_trials : int
        """
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 1_000_000, size=self.n_members)

        shared = dict(self.net_kwargs)
        if optimise and shared.get("hidden_layer_sizes") is None:
            tuner = HeatFluxNet(strategy=self.strategy,
                                random_state=self.random_state, **shared)
            tuner.optimise(X, y, n_trials=n_trials, verbose=False)
            shared["hidden_layer_sizes"] = tuner.hidden_layer_sizes
            shared["activation"] = tuner.activation
            shared["alpha"] = tuner.alpha
            shared["learning_rate_init"] = tuner.learning_rate_init

        self.members_ = []
        for seed in seeds:
            net = HeatFluxNet(strategy=self.strategy, random_state=int(seed),
                              **shared)
            net.fit(X, y, n_samples=n_samples)
            self.members_.append(net)
        self._fitted = True
        return self

    def _member_scalars(self, X: np.ndarray) -> np.ndarray:
        """Each member's scalar heat-flux prediction -> [n_members, n_samples]."""
        self._require_fit()
        rows = []
        for net in self.members_:
            pred = net.predict(X)
            rows.append(pred.sum(axis=1) if pred.ndim == 2 else pred)
        return np.stack(rows, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble mean prediction, shape [n_samples]."""
        return self._member_scalars(X).mean(axis=0)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Ensemble standard deviation (uncertainty), shape [n_samples]."""
        return self._member_scalars(X).std(axis=0)

    def predict_interval(
        self, X: np.ndarray, coverage: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean and a central prediction interval at the requested coverage.

        Returns
        -------
        (mean, lower, upper) : tuple of np.ndarray, each [n_samples]
            Gaussian interval from the ensemble spread:
            ``mean ± z(coverage) · std``.
        """
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1).")
        from scipy.stats import norm

        scalars = self._member_scalars(X)
        mean = scalars.mean(axis=0)
        std = scalars.std(axis=0)
        z = float(norm.ppf(0.5 + coverage / 2))
        return mean, mean - z * std, mean + z * std

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Ensemble has not been fitted. Call fit() first.")
