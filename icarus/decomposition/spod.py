"""
icarus.decomposition.spod
=============================
Spectral Proper Orthogonal Decomposition (SPOD) via Welch's method.

Where standard :class:`~icarus.decomposition.pod.POD` mixes all temporal
frequencies into each energy-ranked mode, SPOD produces modes that are each
**coherent at a single frequency**. This separates the spatial structures by
their timescale — for boiling that means nucleation, bubble departure and
microlayer dynamics can be isolated by the frequency band they live in,
rather than blended together in one POD mode.

Method (Towne, Schmidt & Colonius, JFM 2018)
--------------------------------------------
1. Split the (mean-centred) time series into ``n_blocks`` overlapping blocks.
2. Window each block (Hann) and FFT along time.
3. At each frequency ``f`` assemble the matrix of Fourier realisations across
   blocks, ``Q_f`` of shape [n_pix, n_blocks].
4. The SPOD modes at ``f`` are the eigenvectors of the cross-spectral density
   ``Q_f Q_f^*`` (obtained here from the SVD of ``Q_f``); the eigenvalues are
   the modal energies at that frequency.

The result is, per frequency: an ordered set of orthonormal complex spatial
modes and their energies.

References
----------
Towne, A., Schmidt, O. T. & Colonius, T. (2018). "Spectral proper orthogonal
decomposition and its relationship to dynamic mode decomposition and resolvent
analysis." *Journal of Fluid Mechanics*, 847, 821-867.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SPOD:
    """Spectral POD of a snapshot matrix via Welch's method.

    Parameters
    ----------
    n_modes : int or None
        Number of SPOD modes to retain per frequency. ``None`` keeps all
        (= ``n_blocks``).
    block_size : int
        Number of timesteps per Welch block (``nfft``). Controls the
        frequency resolution: ``df = 1 / (block_size * dt)``. Must be ``<= nt``.
    overlap : float
        Fractional overlap between consecutive blocks, in ``[0, 1)``.
        0.5 (50 %) is the conventional choice.
    window : {"hann", "none"}
        Tapering window applied to each block before the FFT. Hann reduces
        spectral leakage and is the standard for SPOD.

    Attributes
    ----------
    frequencies_ : np.ndarray, shape [n_freq]
        Frequencies (Hz) of the one-sided spectrum, ``rfftfreq(block_size, dt)``.
    modes_ : np.ndarray, complex, shape [n_freq, n_pix, n_modes_retained_]
        SPOD spatial modes; ``modes_[k, :, 0]`` is the leading mode at
        frequency ``frequencies_[k]``.
    eigenvalues_ : np.ndarray, shape [n_freq, n_modes_retained_]
        Modal energy at each frequency, sorted descending along axis 1.
    n_blocks_ : int
        Number of Welch blocks actually used.
    n_modes_retained_ : int

    Examples
    --------
    >>> from icarus.decomposition.spod import SPOD
    >>> from icarus.data.preprocessor import Preprocessor
    >>>
    >>> pre = Preprocessor()
    >>> out = pre.fit_transform(data)
    >>> X_c = Preprocessor.to_matrix(out["T_c"])      # [n_pix, nt]
    >>>
    >>> spod = SPOD(n_modes=3, block_size=256).fit(X_c, dt=data["dt"])
    >>> f = spod.frequencies_
    >>> energy = spod.spectrum()                       # leading-mode energy vs f
    >>> peak_mode = spod.mode(freq=f[energy.argmax()]) # dominant coherent structure
    """

    def __init__(
        self,
        n_modes: Optional[int] = None,
        block_size: int = 256,
        overlap: float = 0.5,
        window: str = "hann",
    ):
        if block_size < 2:
            raise ValueError("block_size must be >= 2.")
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1).")
        if window not in ("hann", "none"):
            raise ValueError("window must be 'hann' or 'none'.")

        self.n_modes = n_modes
        self.block_size = block_size
        self.overlap = overlap
        self.window = window

        # populated after fit()
        self.frequencies_: Optional[np.ndarray] = None
        self.modes_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.n_blocks_: Optional[int] = None
        self.n_modes_retained_: Optional[int] = None
        self._dt: Optional[float] = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_c: np.ndarray, dt: float = 1.0) -> "SPOD":
        """Compute the SPOD of a mean-centred snapshot matrix.

        Parameters
        ----------
        X_c : np.ndarray, shape [n_pix, nt]
            Mean-centred data matrix (rows = pixels, columns = time).
        dt : float
            Timestep in seconds (sets the physical frequency axis).

        Returns
        -------
        self
        """
        if X_c.ndim != 2:
            raise ValueError(f"X_c must be 2-D [n_pix, nt], got shape {X_c.shape}.")
        n_pix, nt = X_c.shape
        nfft = self.block_size
        if nfft > nt:
            raise ValueError(
                f"block_size ({nfft}) must be <= nt ({nt}). "
                "Use a smaller block_size for short records."
            )
        self._dt = float(dt)

        # ── Block segmentation (Welch) ────────────────────────────────────────
        n_ovlp = int(self.overlap * nfft)
        step = nfft - n_ovlp
        n_blocks = 1 + (nt - nfft) // step
        if n_blocks < 1:
            raise ValueError("Not enough timesteps for a single block.")
        self.n_blocks_ = n_blocks

        # ── Window + per-block FFT ────────────────────────────────────────────
        if self.window == "hann":
            win = np.hanning(nfft)
        else:
            win = np.ones(nfft)
        # Energy-preserving scaling: normalise by the window power and block
        # count so eigenvalues are comparable across configurations.
        win_norm = np.sum(win**2)
        scale = np.sqrt(self._dt / (win_norm * n_blocks))

        freqs = np.fft.rfftfreq(nfft, d=self._dt)
        n_freq = freqs.size

        # Q_hat[f, pix, block] — windowed Fourier realisations
        Q_hat = np.empty((n_freq, n_pix, n_blocks), dtype=np.complex128)
        for b in range(n_blocks):
            s = b * step
            block = X_c[:, s:s + nfft] * win[None, :]      # [n_pix, nfft]
            Q_hat[:, :, b] = np.fft.rfft(block, axis=1).T * scale

        # ── Per-frequency SVD → SPOD modes + energies ─────────────────────────
        retained = self.n_modes or n_blocks
        retained = min(retained, n_blocks, n_pix)
        self.n_modes_retained_ = retained

        modes = np.zeros((n_freq, n_pix, retained), dtype=np.complex128)
        eigs = np.zeros((n_freq, retained))
        for k in range(n_freq):
            # SVD of Q_f [n_pix, n_blocks]: left vectors are SPOD modes,
            # singular values squared are the modal energies (eigenvalues of
            # the cross-spectral density Q_f Q_f^*).
            Uf, Sf, _ = np.linalg.svd(Q_hat[k], full_matrices=False)
            modes[k] = Uf[:, :retained]
            eigs[k, :Sf[:retained].size] = (Sf[:retained] ** 2)

        self.frequencies_ = freqs
        self.modes_ = modes
        self.eigenvalues_ = eigs
        return self

    def spectrum(self, mode: int = 0) -> np.ndarray:
        """Energy of a given SPOD mode across frequency.

        Parameters
        ----------
        mode : int
            Mode rank (0 = leading mode at each frequency).

        Returns
        -------
        np.ndarray, shape [n_freq]
            ``eigenvalues_[:, mode]`` — useful for spotting which frequencies
            carry coherent energy (peaks = dominant timescales).
        """
        self._require_fit()
        return self.eigenvalues_[:, mode]

    def mode(self, freq: float, mode: int = 0) -> np.ndarray:
        """Return the SPOD spatial mode nearest a requested frequency.

        Parameters
        ----------
        freq : float
            Target frequency in Hz (snapped to the nearest resolved bin).
        mode : int
            Mode rank (0 = leading).

        Returns
        -------
        np.ndarray, complex, shape [n_pix]
        """
        self._require_fit()
        k = int(np.argmin(np.abs(self.frequencies_ - freq)))
        return self.modes_[k, :, mode]

    def total_energy(self) -> np.ndarray:
        """Total coherent energy (summed over modes) at each frequency.

        Returns
        -------
        np.ndarray, shape [n_freq]
        """
        self._require_fit()
        return self.eigenvalues_.sum(axis=1)

    # ── private helpers ───────────────────────────────────────────────────────

    def _require_fit(self) -> None:
        if self.modes_ is None:
            raise RuntimeError("SPOD has not been fitted. Call fit() first.")
