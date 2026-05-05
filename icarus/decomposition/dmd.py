"""
icarus.decomposition.dmd
============================
Dynamic Mode Decomposition (DMD) for short-horizon heat flux forecasting.

DMD assumes consecutive snapshot pairs are related by a linear operator A::

    X2 ≈ A X1

The reduced operator is found via SVD truncation, and the DMD modes,
eigenvalues, and continuous-time frequencies are extracted from its
eigen-decomposition.

Forecasting uses the DMD expansion::

    x(t) ≈ Σ_k  φ_k · b_k · exp(ω_k · t)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import svd as scipy_svd


class DMD:
    """Dynamic Mode Decomposition with SVD truncation.

    Parameters
    ----------
    energy_threshold : float
        Fraction of energy (0, 1] used to determine SVD truncation rank.
        Default is 0.99 (99 % of energy).
    dt : float
        Timestep in seconds. Used to compute continuous-time frequencies.

    Attributes
    ----------
    modes_ : np.ndarray, shape [n_pix, r]
        DMD spatial modes φ_k.
    eigenvalues_ : np.ndarray, shape [r]
        Discrete-time eigenvalues λ_k.
    frequencies_ : np.ndarray, shape [r]
        Continuous-time frequencies ω_k = ln(λ_k) / dt.
    amplitudes_ : np.ndarray, shape [r]
        Mode amplitudes b_k fitted to the initial condition.

    Examples
    --------
    >>> from icarus.decomposition.dmd import DMD
    >>> dmd = DMD(energy_threshold=0.99, dt=2.5e-4)
    >>> dmd.fit(X_c_train)                          # [n_pix, nt_train]
    >>> X_forecast = dmd.forecast(n_steps=1200)     # [n_pix, 1200]
    """

    def __init__(
        self,
        energy_threshold: float = 0.99,
        dt: float = 1.0,
    ):
        if not (0 < energy_threshold <= 1):
            raise ValueError("energy_threshold must be in (0, 1].")
        self.energy_threshold = energy_threshold
        self.dt = dt

        self.modes_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.frequencies_: Optional[np.ndarray] = None
        self.amplitudes_: Optional[np.ndarray] = None
        self._n_pix: Optional[int] = None
        self._rank: Optional[int] = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_c: np.ndarray) -> "DMD":
        """Fit the DMD model to a mean-centred snapshot matrix.

        Parameters
        ----------
        X_c : np.ndarray, shape [n_pix, nt]
            Mean-centred training data.

        Returns
        -------
        self
        """
        if X_c.ndim != 2:
            raise ValueError(f"X_c must be 2-D [n_pix, nt], got {X_c.shape}.")

        self._n_pix = X_c.shape[0]

        # Build snapshot pairs
        X1 = X_c[:, :-1]
        X2 = X_c[:, 1:]

        # SVD of X1 with energy-based truncation
        U, S, Vt = scipy_svd(X1, full_matrices=False, check_finite=False)
        cum_energy = np.cumsum(S**2) / (S**2).sum()
        r = int(np.searchsorted(cum_energy, self.energy_threshold)) + 1
        self._rank = r

        Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]

        # Reduced DMD operator
        A_tilde = Ur.T @ X2 @ Vtr.T @ np.diag(1.0 / Sr)

        # Eigen-decomposition
        eigenvalues, W = np.linalg.eig(A_tilde)

        # DMD modes (high-dimensional)
        Phi = X2 @ Vtr.T @ np.diag(1.0 / Sr) @ W

        # Continuous-time frequencies
        omega = np.log(eigenvalues) / self.dt

        # Amplitudes: fit to initial condition via pseudo-inverse
        x0 = X_c[:, 0]
        b = np.linalg.lstsq(Phi, x0, rcond=None)[0]

        self.modes_ = Phi
        self.eigenvalues_ = eigenvalues
        self.frequencies_ = omega
        self.amplitudes_ = b

        return self

    def forecast(
        self,
        n_steps: int,
        x_mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forecast the state field forward in time from the initial condition.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to forecast.
        x_mean : np.ndarray, shape [n_pix], optional
            Mean field to add back to the mean-subtracted forecast.

        Returns
        -------
        np.ndarray, shape [n_pix, n_steps]
            Forecasted (mean-added if x_mean given) field.
        """
        self._require_fit()

        k = np.arange(n_steps)
        time_dynamics = self.amplitudes_[:, None] * np.exp(
            np.outer(self.frequencies_, k * self.dt)
        )
        X_forecast = np.real(self.modes_ @ time_dynamics)

        if x_mean is not None:
            X_forecast += x_mean[:, None]

        return X_forecast

    def forecast_from(
        self,
        x_init: np.ndarray,
        n_steps: int,
        x_mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forecast starting from a specific initial state vector.

        This is more accurate than :meth:`forecast` when you want to
        initialise from a specific frame rather than the training start.

        Parameters
        ----------
        x_init : np.ndarray, shape [n_pix]
            Mean-centred initial state (e.g. the last training frame).
        n_steps : int
            Number of steps to forecast.
        x_mean : np.ndarray, optional
            Mean field to add back.

        Returns
        -------
        np.ndarray, shape [n_pix, n_steps]
        """
        self._require_fit()

        b = np.linalg.lstsq(self.modes_, x_init, rcond=None)[0]
        k = np.arange(n_steps)
        time_dynamics = b[:, None] * np.exp(
            np.outer(self.frequencies_, k * self.dt)
        )
        X_forecast = np.real(self.modes_ @ time_dynamics)

        if x_mean is not None:
            X_forecast += x_mean[:, None]

        return X_forecast

    @property
    def growth_rates(self) -> np.ndarray:
        """Real part of continuous-time frequencies (growth / decay rates)."""
        self._require_fit()
        return np.real(self.frequencies_)

    @property
    def oscillation_frequencies(self) -> np.ndarray:
        """Imaginary part of continuous-time frequencies (Hz)."""
        self._require_fit()
        return np.imag(self.frequencies_) / (2 * np.pi)

    # ── private helpers ───────────────────────────────────────────────────────

    def _require_fit(self) -> None:
        if self.modes_ is None:
            raise RuntimeError("DMD has not been fitted. Call fit() first.")
