"""
icarus.pipeline.bandwise
============================
Band-wise modal mapping — "Model C, per timescale".

Standard Model C learns a single POD modal map from temperature to heat flux,
blending all timescales. This model instead:

1. Partitions both temperature and heat flux into the same frequency bands
   (slow convection, bubble departure, fast microlayer) via
   :func:`~icarus.features.partition.partition_by_frequency`;
2. Learns a separate Model-C POD modal mapping **within each band**; and
3. Sums the per-band predictions back into a full heat-flux field.

The motivation: the temperature->heat-flux coupling is likely different per
mechanism, so learning it per timescale is more physical and may generalise
better across surfaces/fluids than one blended mapping. SPOD is the natural way
to choose the band edges (energy peaks = the dominant timescales).

Each band's mapping reuses the standard, tested Model C machinery (POD +
HeatFluxNet), so within-band the per-timestep modal coefficients POD provides
are exactly what the network regresses.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from icarus.decomposition.pod import POD
from icarus.features.engineer import build_modal_features
from icarus.features.partition import partition_by_frequency
from icarus.models.neural import HeatFluxNet
from icarus.models.probabilistic import ProbabilisticHeatFluxNet
from icarus.metrics.evaluation import evaluate


class BandwiseModalModel:
    """Per-band POD modal mapping from temperature to heat flux.

    Parameters
    ----------
    edges : list[float] or "auto"
        Interior frequency band boundaries in Hz (e.g. ``[200, 1000]`` -> three
        bands). Pass ``"auto"`` to derive them from the data during ``fit``:
        SPOD is run on the heat flux and each edge is placed at the energy
        valley between adjacent spectral peaks
        (:meth:`SPOD.suggest_band_edges`).
    n_bands : int
        Number of bands when ``edges="auto"`` (ignored otherwise).
    n_pod_modes : int
        POD modes retained per band, per field.
    model_kwargs : dict, optional
        Extra keyword arguments forwarded to each band's :class:`HeatFluxNet`
        (e.g. ``hidden_layer_sizes``, ``max_iter``).
    optimise_hyperparams : bool
        Run Optuna per band before fitting (slower).
    n_trials : int
    n_training_samples : int or None
    n_members : int
        If > 1, each band is a deep ensemble of this many networks, enabling
        :meth:`predict_interval` (mean + uncertainty). 1 = a single network.
    random_state : int

    Examples
    --------
    >>> import icarus as tf
    >>> T, _ = tf.load_field("MODEL_~1.MAT", "temp")
    >>> q, dt = tf.load_field("MODEL_~1.MAT", "heatflux")
    >>> model = tf.BandwiseModalModel(edges=[200, 1000], n_pod_modes=5, n_members=5)
    >>> model.fit(T, q, dt=dt, spatial_crop=5, trim_frames=43)
    >>> q_pred = model.predict(T)
    >>> mean, lower, upper = model.predict_interval(T, coverage=0.9)
    >>> model.evaluate()        # total + per-band R²
    """

    def __init__(
        self,
        edges,
        n_bands: int = 3,
        n_pod_modes: int = 5,
        model_kwargs: Optional[dict] = None,
        optimise_hyperparams: bool = False,
        n_trials: int = 20,
        n_training_samples: Optional[int] = None,
        n_members: int = 1,
        random_state: int = 42,
    ):
        self.edges = edges
        self.n_bands = n_bands
        self.n_pod_modes = n_pod_modes
        self.model_kwargs = model_kwargs or {"hidden_layer_sizes": (64,)}
        self.optimise_hyperparams = optimise_hyperparams
        self.n_trials = n_trials
        self.n_training_samples = n_training_samples
        self.n_members = n_members
        self.random_state = random_state

        self.bands_: dict[str, dict] = {}
        self.labels_: list[str] = []
        self.q_mean_field_: Optional[np.ndarray] = None
        self.dt_: Optional[float] = None
        self._ny: Optional[int] = None
        self._nx: Optional[int] = None
        self._T_train: Optional[np.ndarray] = None
        self._q_train: Optional[np.ndarray] = None
        self._fitted = False

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        T: np.ndarray,
        q: np.ndarray,
        dt: float,
        spatial_crop: int = 0,
        trim_frames: int = 0,
        verbose: bool = False,
    ) -> "BandwiseModalModel":
        """Fit one Model-C mapping per frequency band.

        Parameters
        ----------
        T, q : np.ndarray, shape [ny, nx, nt]
            Temperature and heat-flux fields.
        dt : float
            Timestep in seconds.
        spatial_crop, trim_frames : int
            Border pixels / startup frames to drop from both fields.
        """
        T = self._crop_trim(T, spatial_crop, trim_frames)
        q = self._crop_trim(q, spatial_crop, trim_frames)
        if T.shape != q.shape:
            raise ValueError(f"T {T.shape} and q {q.shape} must match.")

        self._ny, self._nx, _ = T.shape
        self.dt_ = float(dt)
        self._T_train, self._q_train = T, q

        if isinstance(self.edges, str):
            if self.edges != "auto":
                raise ValueError(f"edges must be a list or 'auto', got {self.edges!r}.")
            self.edges = self._auto_edges(q, dt, verbose=verbose)

        T_parts = partition_by_frequency(T, dt, self.edges)
        q_parts = partition_by_frequency(q, dt, self.edges)
        self.labels_ = T_parts.labels
        self.q_mean_field_ = q_parts.mean_field

        for label in self.labels_:
            if verbose:
                print(f"[bandwise] fitting band {label} ...")
            self.bands_[label] = self._fit_band(
                T_parts.components[label], q_parts.components[label])

        self._fitted = True
        return self

    def predict(self, T_new: np.ndarray) -> np.ndarray:
        """Predict the full heat-flux field from a temperature field.

        Returns
        -------
        np.ndarray, shape [ny, nx, nt]
        """
        self._require_fit()
        ny, nx = self._ny, self._nx
        nt = T_new.shape[2]

        fluctuation = np.zeros((ny * nx, nt))
        T_parts = partition_by_frequency(T_new, self.dt_, self.edges)
        for label in self.labels_:
            fluctuation += self._predict_band_flat(label, T_parts.components[label])

        absolute = fluctuation + self.q_mean_field_.reshape(-1)[:, None]
        return absolute.reshape(ny, nx, nt)

    def predict_interval(
        self, T_new: np.ndarray, coverage: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the heat-flux field with a calibrated uncertainty band.

        Requires ``n_members > 1``. Each band contributes an independent
        ensemble mean and variance; the band means sum and the band variances
        add (bands treated as independent), giving a total mean and a Gaussian
        interval at the requested coverage.

        Returns
        -------
        (mean, lower, upper) : tuple of np.ndarray, each [ny, nx, nt]
        """
        self._require_fit()
        if self.n_members <= 1:
            raise RuntimeError(
                "predict_interval needs n_members > 1; refit with an ensemble.")
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1).")
        from scipy.stats import norm

        ny, nx = self._ny, self._nx
        nt = T_new.shape[2]
        T_parts = partition_by_frequency(T_new, self.dt_, self.edges)

        mean = np.zeros((ny * nx, nt))
        variance = np.zeros((ny * nx, nt))
        for label in self.labels_:
            band_field = T_parts.components[label]
            mean += self._predict_band_flat(label, band_field)
            # total = epistemic (ensemble spread) + aleatoric (residual scatter)
            variance += self._predict_band_std(label, band_field) ** 2
            variance += self.bands_[label]["aleatoric_var"]

        mean += self.q_mean_field_.reshape(-1)[:, None]
        half = float(norm.ppf(0.5 + coverage / 2)) * np.sqrt(variance)
        lower, upper = mean - half, mean + half
        return (mean.reshape(ny, nx, nt),
                lower.reshape(ny, nx, nt),
                upper.reshape(ny, nx, nt))

    def evaluate(self, verbose: bool = True) -> dict:
        """Total and per-band R²/RMSE on the training data.

        Returns
        -------
        dict with key ``"total"`` (fluctuation-field metrics) and one entry per
        band label (that band's component metrics).
        """
        self._require_fit()
        ny, nx = self._ny, self._nx
        nt = self._T_train.shape[2]

        q_parts = partition_by_frequency(self._q_train, self.dt_, self.edges)
        T_parts = partition_by_frequency(self._T_train, self.dt_, self.edges)

        results: dict = {}
        total_true = np.zeros(ny * nx * nt)
        total_pred = np.zeros(ny * nx * nt)
        for label in self.labels_:
            true_flat = q_parts.components[label].reshape(ny * nx, nt)
            true_flat = (true_flat - true_flat.mean(axis=1, keepdims=True)).reshape(-1)
            pred_flat = self._predict_band_flat(
                label, T_parts.components[label]).reshape(-1)
            results[label] = evaluate(true_flat, pred_flat, split=label)
            total_true += true_flat
            total_pred += pred_flat

        results["total"] = evaluate(total_true, total_pred, split="total")
        if verbose:
            print(results["total"])
            for label in self.labels_:
                print(results[label])
        return results

    # ── per-band helpers ──────────────────────────────────────────────────────

    def _auto_edges(self, q: np.ndarray, dt: float, verbose: bool) -> list[float]:
        """Choose band edges from the heat flux's SPOD spectrum (valleys
        between the ``n_bands`` strongest coherent peaks)."""
        from icarus.decomposition.spod import SPOD

        nt = q.shape[2]
        block = max(64, min(1024, nt // 2))
        spod = SPOD(n_modes=2, block_size=block).fit(self._centre_matrix(q), dt=dt)
        edges = spod.suggest_band_edges(n_bands=self.n_bands)
        if verbose:
            print(f"[bandwise] auto edges from SPOD: {edges} Hz")
        return edges

    def _fit_band(self, T_band: np.ndarray, q_band: np.ndarray) -> dict:
        X_c_T = self._centre_matrix(T_band)
        X_c_q = self._centre_matrix(q_band)

        pod_T = POD(n_modes=self.n_pod_modes).fit(X_c_T)
        pod_q = POD(n_modes=self.n_pod_modes).fit(X_c_q)
        T_contribs = pod_T.modal_contributions(X_c_T)
        q_contribs = pod_q.modal_contributions(X_c_q)
        X, y = build_modal_features(T_contribs, q_contribs)

        if self.n_members > 1:
            net = ProbabilisticHeatFluxNet(
                n_members=self.n_members, strategy="modal",
                random_state=self.random_state, **self.model_kwargs)
            net.fit(X, y, n_samples=self.n_training_samples,
                    optimise=self.optimise_hyperparams, n_trials=self.n_trials)
        else:
            net = HeatFluxNet(strategy="modal", random_state=self.random_state,
                              **self.model_kwargs)
            if self.optimise_hyperparams and net.hidden_layer_sizes is None:
                net.optimise(X, y, n_trials=self.n_trials, verbose=False)
            net.fit(X, y, n_samples=self.n_training_samples)

        # Aleatoric variance: everything the modal model cannot reproduce of
        # the TRUE band field — both network error and the incoherent scatter
        # that POD truncation discards (the noise lives in the dropped modes,
        # so it must be measured against the true field, not the modal target).
        # A deep ensemble captures only epistemic (model-disagreement)
        # uncertainty; this aleatoric term captures the data randomness, which
        # dominates for boiling. Total predictive variance = epistemic +
        # aleatoric.
        pred = net.predict(X)
        pred_scalar = pred.sum(axis=1) if pred.ndim == 2 else pred
        true_scalar = X_c_q.T.reshape(-1)          # true band field, time-major
        residual = true_scalar - pred_scalar
        aleatoric_var = float(np.mean(residual ** 2))
        return {"pod_T": pod_T, "pod_q": pod_q, "net": net,
                "aleatoric_var": aleatoric_var}

    def _band_features(self, label: str, T_band_new: np.ndarray):
        """Modal feature matrix for a band's new temperature data."""
        band = self.bands_[label]
        X_c_T = self._centre_matrix(T_band_new)
        T_contribs = band["pod_T"].modal_contributions(X_c_T)
        X, _ = build_modal_features(T_contribs)         # time-major rows
        return band, X

    def _to_field(self, flat: np.ndarray, nt: int) -> np.ndarray:
        """Un-flatten a time-major scalar vector to [n_pix, nt]."""
        return flat.reshape(nt, self._ny * self._nx).T

    def _predict_band_flat(self, label: str, T_band_new: np.ndarray) -> np.ndarray:
        """Return the band's predicted heat-flux fluctuation mean as [n_pix, nt]."""
        band, X = self._band_features(label, T_band_new)
        net = band["net"]
        if isinstance(net, ProbabilisticHeatFluxNet):
            flat = net.predict(X)                       # already summed scalar
        else:
            pred = net.predict(X)
            flat = pred.sum(axis=1) if pred.ndim == 2 else pred
        return self._to_field(flat, T_band_new.shape[2])

    def _predict_band_std(self, label: str, T_band_new: np.ndarray) -> np.ndarray:
        """Return the band's ensemble std (uncertainty) as [n_pix, nt]."""
        band, X = self._band_features(label, T_band_new)
        net = band["net"]
        if not isinstance(net, ProbabilisticHeatFluxNet):
            raise RuntimeError(
                "predict_interval needs n_members > 1 (a band ensemble).")
        return self._to_field(net.predict_std(X), T_band_new.shape[2])

    @staticmethod
    def _crop_trim(field: np.ndarray, spatial_crop: int, trim_frames: int) -> np.ndarray:
        if spatial_crop > 0:
            field = field[spatial_crop:-spatial_crop, spatial_crop:-spatial_crop, :]
        if trim_frames > 0:
            field = field[:, :, trim_frames:]
        return field

    @staticmethod
    def _centre_matrix(field: np.ndarray) -> np.ndarray:
        ny, nx, nt = field.shape
        X = field.reshape(ny * nx, nt)
        return X - X.mean(axis=1, keepdims=True)

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
