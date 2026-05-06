"""
icarus.decomposition.pod
============================
Proper Orthogonal Decomposition (POD) via the method of snapshots (SVD).

The mean-centred data matrix  X_c  is factorised as::

    X_c = U Σ V^T

where

- ``U``  [n_pix × r]  – orthonormal spatial modes
- ``Σ``  [r × r]      – diagonal matrix of singular values
- ``V^T``[r × nt]     – temporal coefficients

The modal energy fraction and cumulative energy are derived from the
singular values, and the rank-1 modal contributions are::

    C_i(x, t) = U_i * σ_i * V_i^T

These contributions are the features used by Model C in the paper.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import svd as scipy_svd


class POD:
    """Proper Orthogonal Decomposition of a snapshot matrix.

    Parameters
    ----------
    n_modes : int or None
        Number of modes to retain.  If ``None``, retain all modes.
    energy_threshold : float or None
        If set, retain the minimum number of modes that captures this
        fraction of the total fluctuation energy (e.g. ``0.99`` for 99 %).
        Overrides ``n_modes`` when both are given.

    Attributes
    ----------
    modes_ : np.ndarray, shape [n_pix, r]
        Spatial POD modes (columns of U).
    singular_values_ : np.ndarray, shape [r]
        Singular values σ_i.
    temporal_coefficients_ : np.ndarray, shape [r, nt]
        A = Σ V^T  –  amplitude of each mode at each timestep.
    energy_fractions_ : np.ndarray, shape [r]
        Fraction of total variance captured by each mode.
    cumulative_energy_ : np.ndarray, shape [r]
        Cumulative energy fraction.
    n_modes_retained_ : int
        Number of modes actually retained after fitting.

    Examples
    --------
    >>> from icarus.decomposition.pod import POD
    >>> from icarus.data.preprocessor import Preprocessor
    >>>
    >>> pre = Preprocessor()
    >>> out = pre.fit_transform(data)
    >>> X_c = Preprocessor.to_matrix(out["T_c"])   # [n_pix, nt]
    >>>
    >>> pod = POD(n_modes=10)
    >>> pod.fit(X_c)
    >>> print(pod.cumulative_energy_[:5])           # first 5 modes
    >>> contribs = pod.modal_contributions(X_c)    # [n_pix, nt, n_modes]
    """

    def __init__(
        self,
        n_modes: Optional[int] = None,
        energy_threshold: Optional[float] = None,
    ):
        if energy_threshold is not None and not (0 < energy_threshold <= 1):
            raise ValueError("energy_threshold must be in (0, 1].")

        self.n_modes = n_modes
        self.energy_threshold = energy_threshold

        # populated after fit()
        self.modes_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.temporal_coefficients_: Optional[np.ndarray] = None
        self.energy_fractions_: Optional[np.ndarray] = None
        self.cumulative_energy_: Optional[np.ndarray] = None
        self.n_modes_retained_: Optional[int] = None
        self._n_pix: Optional[int] = None
        self._nt: Optional[int] = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_c: np.ndarray) -> "POD":
        """Decompose the mean-centred snapshot matrix.

        Parameters
        ----------
        X_c : np.ndarray, shape [n_pix, nt]
            Mean-centred data matrix from
            :meth:`~icarus.data.preprocessor.Preprocessor.to_matrix`.

        Returns
        -------
        self
        """
        if X_c.ndim != 2:
            raise ValueError(f"X_c must be 2-D [n_pix, nt], got shape {X_c.shape}.")

        self._n_pix, self._nt = X_c.shape

        # Economy SVD: U [n_pix × k], S [k], Vt [k × nt]
        U, S, Vt = scipy_svd(X_c, full_matrices=False, check_finite=False)

        # Energy fractions
        energy = S**2
        total_energy = energy.sum()
        self.energy_fractions_ = energy / total_energy
        self.cumulative_energy_ = np.cumsum(self.energy_fractions_)

        # Determine rank
        r = self._determine_rank(S)
        self.n_modes_retained_ = r

        # Store truncated decomposition
        self.modes_ = U[:, :r]
        self.singular_values_ = S[:r]
        # Temporal coefficients: A = Σ V^T  [r × nt]
        self.temporal_coefficients_ = np.diag(S[:r]) @ Vt[:r, :]

        return self

    def modal_contributions(
        self,
        X_c: np.ndarray,
        n_modes: Optional[int] = None,
    ) -> np.ndarray:
        """Compute rank-1 modal contribution fields.

        Each contribution  C_i(x, t) = φ_i(x) · σ_i · v_i(t)  is the
        spatial-temporal field explained by mode i alone.

        Parameters
        ----------
        X_c : np.ndarray, shape [n_pix, nt]
            Mean-centred snapshot matrix.
        n_modes : int, optional
            Override the number of modes to compute contributions for.

        Returns
        -------
        np.ndarray, shape [n_pix, nt, n_modes]
            Modal contribution fields.
        """
        self._require_fit()
        r = n_modes or self.n_modes_retained_
        r = min(r, self.n_modes_retained_)

        # Project X_c onto fitted basis — handles nt_in != nt_fit
        nt_in = X_c.shape[1]
        # U^T @ X_c gives Σ V^T directly — no further σ scaling needed
        A_in = self.modes_[:, :r].T @ X_c
        contribs = np.zeros((self._n_pix, nt_in, r))
        for i in range(r):
            contribs[:, :, i] = np.outer(self.modes_[:, i], A_in[i, :])
        return contribs

    def reconstruct(self, n_modes: Optional[int] = None) -> np.ndarray:
        """Reconstruct the mean-subtracted field from the first k modes.

        Parameters
        ----------
        n_modes : int, optional
            Number of modes to use. Defaults to ``n_modes_retained_``.

        Returns
        -------
        np.ndarray, shape [n_pix, nt]
        """
        self._require_fit()
        r = n_modes or self.n_modes_retained_
        r = min(r, self.n_modes_retained_)
        return self.modes_[:, :r] @ self.temporal_coefficients_[:r, :]

    def project(self, X_c_new: np.ndarray) -> np.ndarray:
        """Project new data onto the fitted POD basis.

        Useful for computing temporal coefficients on unseen data
        (e.g. a test set) without refitting.

        Parameters
        ----------
        X_c_new : np.ndarray, shape [n_pix, nt_new]

        Returns
        -------
        np.ndarray, shape [n_modes_retained_, nt_new]
            Temporal coefficients on the new data.
        """
        self._require_fit()
        # U^T @ X_c_new gives projected temporal coefficients
        return self.modes_.T @ X_c_new

    def modes_needed_for_energy(self, threshold: float) -> int:
        """Return the number of modes needed to capture a given energy fraction.

        Parameters
        ----------
        threshold : float
            Energy fraction in (0, 1], e.g. 0.90 for 90 %.

        Returns
        -------
        int
        """
        self._require_fit()
        idx = np.searchsorted(self.cumulative_energy_, threshold)
        return int(idx) + 1

    # ── private helpers ───────────────────────────────────────────────────────

    def _determine_rank(self, S: np.ndarray) -> int:
        if self.energy_threshold is not None:
            cum = np.cumsum(S**2) / (S**2).sum()
            r = int(np.searchsorted(cum, self.energy_threshold)) + 1
        elif self.n_modes is not None:
            r = min(self.n_modes, len(S))
        else:
            r = len(S)
        return r

    def _require_fit(self) -> None:
        if self.modes_ is None:
            raise RuntimeError("POD has not been fitted. Call fit() first.")
