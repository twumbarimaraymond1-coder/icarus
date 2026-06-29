"""
icarus.features.partition
=============================
Frequency partitioning of a heat-flux (or temperature) field.

A data-driven stand-in for the classical physical heat-flux partition
(convection / evaporation / quenching): each pixel's time signal is split into
additive frequency bands, so slow large-scale behaviour, bubble-departure
oscillations, and fast microlayer dynamics land in separate component fields.

The components are **exactly additive** — they sum back to the mean-subtracted
field — which is the defining property of a partition. Band edges can be chosen
from an :class:`~icarus.decomposition.spod.SPOD` energy spectrum (peaks = the
natural timescales), or set physically once bubble-departure frequencies are
known.

If, later, true physical component targets become available (e.g. from
synchronised high-speed video), they can be dropped in wherever these
components are used — the downstream prediction framework is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Optional

import numpy as np


@dataclass
class Partition:
    """Result of :func:`partition_by_frequency`.

    Attributes
    ----------
    components : dict[str, np.ndarray]
        Band label -> component field ``[ny, nx, nt]``. The components sum to
        the mean-subtracted input field.
    mean_field : np.ndarray, shape [ny, nx]
        Per-pixel time mean removed before partitioning (add back to rebuild
        the absolute field).
    edges : list[float]
        Interior band boundaries in Hz.
    dt : float
        Timestep used (seconds).
    """

    components: dict[str, np.ndarray]
    mean_field: np.ndarray
    edges: list[float] = _field(default_factory=list)
    dt: float = 1.0

    @property
    def labels(self) -> list[str]:
        return list(self.components.keys())

    def energy(self) -> dict[str, float]:
        """Fraction of total fluctuation energy carried by each band.

        Returns
        -------
        dict[str, float]
            Band label -> energy fraction (sums to ~1).
        """
        powers = {k: float(np.sum(v.astype(np.float64) ** 2))
                  for k, v in self.components.items()}
        total = sum(powers.values()) or 1.0
        return {k: p / total for k, p in powers.items()}

    def total(self) -> np.ndarray:
        """Reconstruct the original field: mean + sum of components."""
        summed = sum(self.components.values())
        return summed + self.mean_field[:, :, None]


def partition_by_frequency(
    field: np.ndarray,
    dt: float,
    edges: list[float],
    labels: Optional[list[str]] = None,
) -> Partition:
    """Split a ``[ny, nx, nt]`` field into additive frequency-band components.

    Each pixel's time series is Fourier transformed, sliced into the bands
    defined by ``edges``, and inverse transformed. The resulting component
    fields sum exactly to the mean-subtracted input.

    Parameters
    ----------
    field : np.ndarray, shape [ny, nx, nt]
    dt : float
        Timestep in seconds (sets the frequency axis).
    edges : list[float]
        Interior band boundaries in Hz, ascending. ``[200, 1000]`` produces
        three bands: ``0-200``, ``200-1000``, ``1000+`` Hz.
    labels : list[str], optional
        Names for the bands (len = ``len(edges) + 1``). Defaults to readable
        frequency ranges.

    Returns
    -------
    Partition
    """
    if field.ndim != 3:
        raise ValueError(f"field must be [ny, nx, nt], got {field.shape}.")
    if list(edges) != sorted(edges) or any(e <= 0 for e in edges):
        raise ValueError("edges must be ascending and positive.")

    ny, nx, nt = field.shape
    snapshots = field.reshape(ny * nx, nt)
    mean = snapshots.mean(axis=1)
    centred = snapshots - mean[:, None]

    spectrum = np.fft.rfft(centred, axis=1)
    freqs = np.fft.rfftfreq(nt, d=dt)

    bounds = [0.0, *edges, np.inf]
    if labels is None:
        labels = _default_labels(edges)
    if len(labels) != len(bounds) - 1:
        raise ValueError(
            f"labels must have {len(bounds) - 1} entries, got {len(labels)}.")

    components: dict[str, np.ndarray] = {}
    for i, name in enumerate(labels):
        lo, hi = bounds[i], bounds[i + 1]
        band_mask = (freqs >= lo) & (freqs < hi)
        band_spectrum = spectrum * band_mask
        component = np.fft.irfft(band_spectrum, n=nt, axis=1)
        components[name] = component.reshape(ny, nx, nt)

    return Partition(
        components=components,
        mean_field=mean.reshape(ny, nx),
        edges=list(edges),
        dt=float(dt),
    )


def _default_labels(edges: list[float]) -> list[str]:
    bounds = [0.0, *edges]
    labels = []
    for i, lo in enumerate(bounds):
        if i + 1 < len(bounds):
            labels.append(f"{lo:g}-{bounds[i + 1]:g} Hz")
        else:
            labels.append(f"{lo:g}+ Hz")
    return labels
