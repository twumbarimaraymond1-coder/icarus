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
        if self.block_size > X_c.shape[1]:
            raise ValueError(
                f"block_size ({self.block_size}) must be <= nt ({X_c.shape[1]}). "
                "Use a smaller block_size for short records."
            )
        self._dt = float(dt)

        # Welch: split into overlapping windowed blocks, FFT each in time.
        freqs, Q_hat = self._welch_transform(X_c)
        # Per-frequency SVD gives the coherent spatial modes and their energies.
        modes, eigs = self._decompose_per_frequency(Q_hat, n_pix=X_c.shape[0])

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

    def dominant_frequencies(
        self,
        n: int = 4,
        mode: int = 0,
        ignore_dc: bool = True,
    ) -> np.ndarray:
        """Frequencies (Hz) of the strongest spectral peaks, energy-sorted.

        Finds local maxima in a mode's energy spectrum and returns the
        frequencies of the ``n`` strongest, highest first. These are the
        candidate dominant timescales (e.g. bubble departure) in the data.

        Parameters
        ----------
        n : int
            Maximum number of peaks to return.
        mode : int
            Mode rank whose spectrum is searched (0 = leading).
        ignore_dc : bool
            Skip the zero-frequency bin (the slow mean drift), which would
            otherwise dominate.

        Returns
        -------
        np.ndarray, shape [<= n]
            Peak frequencies in Hz, sorted by descending energy.
        """
        self._require_fit()
        e = self.eigenvalues_[:, mode]
        is_peak = np.zeros(e.shape, dtype=bool)
        is_peak[1:-1] = (e[1:-1] > e[:-2]) & (e[1:-1] > e[2:])
        if ignore_dc:
            is_peak[0] = False
        idx = np.where(is_peak)[0]
        if idx.size == 0:                       # fallback: strongest non-DC bin
            idx = np.array([int(np.argmax(e[1:]) + 1)])
        idx = idx[np.argsort(e[idx])[::-1][:n]]
        return self.frequencies_[idx]

    # ── private helpers ───────────────────────────────────────────────────────

    def _welch_transform(self, X_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment time into overlapping windowed blocks and FFT each.

        Returns the one-sided frequency axis and the energy-normalised
        windowed Fourier realisations ``Q_hat[freq, pix, block]``. Hann
        windowing suppresses spectral leakage; the scaling normalises by window
        power and block count so eigenvalues are comparable across settings.
        """
        n_pix, nt = X_c.shape
        nfft = self.block_size
        step = nfft - int(self.overlap * nfft)
        n_blocks = 1 + (nt - nfft) // step
        if n_blocks < 1:
            raise ValueError("Not enough timesteps for a single block.")
        self.n_blocks_ = n_blocks

        win = np.hanning(nfft) if self.window == "hann" else np.ones(nfft)
        scale = np.sqrt(self._dt / (np.sum(win**2) * n_blocks))
        freqs = np.fft.rfftfreq(nfft, d=self._dt)

        Q_hat = np.empty((freqs.size, n_pix, n_blocks), dtype=np.complex128)
        for b in range(n_blocks):
            s = b * step
            block = X_c[:, s:s + nfft] * win[None, :]       # [n_pix, nfft]
            Q_hat[:, :, b] = np.fft.rfft(block, axis=1).T * scale
        return freqs, Q_hat

    def _decompose_per_frequency(
        self, Q_hat: np.ndarray, n_pix: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """SVD each frequency's realisation matrix into SPOD modes + energies.

        For ``Q_hat[k]`` of shape [n_pix, n_blocks], the left singular vectors
        are the SPOD modes and the squared singular values are the modal
        energies (eigenvalues of the cross-spectral density ``Q_f Q_f^*``).
        """
        n_freq, _, n_blocks = Q_hat.shape
        retained = min(self.n_modes or n_blocks, n_blocks, n_pix)
        self.n_modes_retained_ = retained

        modes = np.zeros((n_freq, n_pix, retained), dtype=np.complex128)
        eigs = np.zeros((n_freq, retained))
        for k in range(n_freq):
            Uf, Sf, _ = np.linalg.svd(Q_hat[k], full_matrices=False)
            modes[k] = Uf[:, :retained]
            eigs[k, :Sf[:retained].size] = Sf[:retained] ** 2
        return modes, eigs

    def _require_fit(self) -> None:
        if self.modes_ is None:
            raise RuntimeError("SPOD has not been fitted. Call fit() first.")
