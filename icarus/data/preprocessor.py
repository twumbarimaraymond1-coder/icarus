"""
icarus.data.preprocessor
============================
Preprocessing for IR thermography datasets prior to decomposition and
machine learning.

Steps performed
---------------
1. Spatial cropping  – removes edge distortions from the heater boundary.
2. Temporal trimming – removes corrupt trailing frames.
3. Mean subtraction  – isolates the fluctuating component for POD/DMD.
4. Reshaping         – converts [ny, nx, nt] fields to [n_pix, nt] matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PreprocessorConfig:
    """Configuration for the Preprocessor.

    Attributes
    ----------
    spatial_crop : int
        Number of pixels to remove from each spatial boundary (default 5).
    trim_frames : int
        Number of trailing time frames to discard (default 0).
    """
    spatial_crop: int = 5
    trim_frames: int = 0


class Preprocessor:
    """Prepare raw IR thermography data for reduced-order analysis.

    Parameters
    ----------
    config : PreprocessorConfig, optional
        Preprocessing settings. Defaults to spatial_crop=5, trim_frames=0.

    Examples
    --------
    >>> from icarus.data.preprocessor import Preprocessor, PreprocessorConfig
    >>> cfg = PreprocessorConfig(spatial_crop=5, trim_frames=43)
    >>> pre = Preprocessor(cfg)
    >>> data_out = pre.fit_transform(data)
    >>> T_matrix = pre.to_matrix(data_out["T"])   # shape [n_pix, nt]
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self._T_mean: Optional[np.ndarray] = None
        self._q_mean: Optional[np.ndarray] = None
        self._fitted = False

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, data: dict[str, np.ndarray]) -> "Preprocessor":
        """Compute mean fields from the data (without transforming).

        Parameters
        ----------
        data : dict
            Output of :func:`icarus.data.loader.load` with keys
            ``"T"``, ``"q"``, ``"dt"``.

        Returns
        -------
        self
        """
        T, q = self._crop_and_trim(data["T"], data["q"])
        self._T_mean = T.mean(axis=2, keepdims=True)
        self._q_mean = q.mean(axis=2, keepdims=True)
        self._fitted = True
        return self

    def transform(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply preprocessing using previously fitted mean fields.

        Parameters
        ----------
        data : dict
            Raw dataset dict.

        Returns
        -------
        dict with keys:

        - ``"T"``       : cropped temperature field [ny, nx, nt]
        - ``"q"``       : cropped heat flux field [ny, nx, nt]
        - ``"T_mean"``  : spatial-temporal mean temperature [ny, nx, 1]
        - ``"q_mean"``  : spatial-temporal mean heat flux [ny, nx, 1]
        - ``"T_c"``     : mean-subtracted temperature [ny, nx, nt]
        - ``"q_c"``     : mean-subtracted heat flux [ny, nx, nt]
        - ``"dt"``      : timestep in seconds
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        T, q = self._crop_and_trim(data["T"], data["q"])
        return {
            "T": T,
            "q": q,
            "T_mean": self._T_mean,
            "q_mean": self._q_mean,
            "T_c": T - self._T_mean,
            "q_c": q - self._q_mean,
            "dt": data["dt"],
        }

    def fit_transform(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Fit and transform in one step.

        Parameters
        ----------
        data : dict
            Raw dataset dict from :func:`icarus.data.loader.load`.

        Returns
        -------
        Preprocessed dataset dict (same keys as :meth:`transform`).

        Examples
        --------
        >>> pre = Preprocessor(PreprocessorConfig(spatial_crop=5, trim_frames=43))
        >>> out = pre.fit_transform(raw_data)
        >>> out["T"].shape        # (41, 106, 4000) for the example dataset
        >>> out["T_c"].shape      # same, mean-subtracted
        """
        return self.fit(data).transform(data)

    @staticmethod
    def to_matrix(field: np.ndarray) -> np.ndarray:
        """Reshape a 3-D field [ny, nx, nt] into a 2-D matrix [n_pix, nt].

        This is the format required by POD and DMD.

        Parameters
        ----------
        field : np.ndarray, shape [ny, nx, nt]

        Returns
        -------
        np.ndarray, shape [ny*nx, nt]
        """
        ny, nx, nt = field.shape
        return field.reshape(ny * nx, nt)

    @staticmethod
    def from_matrix(matrix: np.ndarray, ny: int, nx: int) -> np.ndarray:
        """Reshape a 2-D matrix [n_pix, nt] back to a 3-D field [ny, nx, nt].

        Parameters
        ----------
        matrix : np.ndarray, shape [n_pix, nt]
        ny, nx : int
            Spatial dimensions of the original field.

        Returns
        -------
        np.ndarray, shape [ny, nx, nt]
        """
        nt = matrix.shape[1]
        return matrix.reshape(ny, nx, nt)

    @property
    def T_mean(self) -> np.ndarray:
        self._require_fit()
        return self._T_mean.squeeze()

    @property
    def q_mean(self) -> np.ndarray:
        self._require_fit()
        return self._q_mean.squeeze()

    # ── private helpers ───────────────────────────────────────────────────────

    def _crop_and_trim(
        self,
        T: np.ndarray,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.config.spatial_crop
        tf = self.config.trim_frames

        if c > 0:
            T = T[c:-c, c:-c, :]
            q = q[c:-c, c:-c, :]

        if tf > 0:
            T = T[:, :, :-tf]
            q = q[:, :, :-tf]

        return T, q

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")
