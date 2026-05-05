"""
icarus.visualisation.plots
==============================
Publication-quality plotting utilities for thermography and POD/DMD results.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


# ── colour helpers ────────────────────────────────────────────────────────────

def _diverging_cmap():
    """Blue-white-red diverging colourmap for signed POD modes."""
    return plt.cm.RdBu_r


def _symmetric_clim(data: np.ndarray) -> tuple[float, float]:
    v = np.nanmax(np.abs(data))
    return -v, v


# ── public API ────────────────────────────────────────────────────────────────

def plot_field(
    field: np.ndarray,
    title: str = "",
    cbar_label: str = "Heat flux (W/m²)",
    cmap: str = "inferno",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Figure:
    """Plot a 2-D spatial field (temperature or heat flux at one timestep).

    Parameters
    ----------
    field : np.ndarray, shape [ny, nx]
    title : str
    cbar_label : str
    cmap : str
    ax : matplotlib.Axes, optional
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    im = ax.imshow(field, cmap=cmap, origin="upper", aspect="equal", **kwargs)
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("x pixels")
    ax.set_ylabel("y pixels")
    if standalone:
        plt.tight_layout()
    return fig


def plot_pod_modes(
    pod,
    ny: int,
    nx: int,
    n_modes: int = 5,
    figsize_per_mode: tuple = (3.5, 4),
) -> Figure:
    """Plot spatial POD modes and their temporal coefficients side by side.

    Parameters
    ----------
    pod : POD
        Fitted POD object.
    ny, nx : int
        Spatial dimensions for reshaping modes.
    n_modes : int
        Number of modes to plot.
    """
    n = min(n_modes, pod.n_modes_retained_)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(figsize_per_mode[0] * 2, figsize_per_mode[1] * n),
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    cmap = _diverging_cmap()

    for i in range(n):
        mode_2d = pod.modes_[:, i].reshape(ny, nx)
        vmin, vmax = _symmetric_clim(mode_2d)

        # Spatial mode
        ax_s = axes[i, 0]
        im = ax_s.imshow(mode_2d, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        plt.colorbar(im, ax=ax_s)
        energy_pct = pod.energy_fractions_[i] * 100
        ax_s.set_title(f"Mode {i+1} spatial ({energy_pct:.2f}% energy)")
        ax_s.set_xlabel("x index")
        ax_s.set_ylabel("y index")

        # Temporal coefficient
        ax_t = axes[i, 1]
        ax_t.plot(pod.temporal_coefficients_[i, :], lw=0.8, color="steelblue")
        ax_t.axhline(0, color="k", lw=0.5, ls="--")
        ax_t.set_title(f"Mode {i+1} temporal coefficient")
        ax_t.set_xlabel("Time steps")
        ax_t.set_ylabel("Amplitude")
        ax_t.grid(True, alpha=0.3)

    fig.suptitle("POD modes — spatial structure (left) | temporal coefficient (right)")
    plt.tight_layout()
    return fig


def plot_cumulative_energy(pod, ax: Optional[plt.Axes] = None) -> Figure:
    """Plot the cumulative POD energy distribution.

    Parameters
    ----------
    pod : POD
        Fitted POD object.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    modes = np.arange(1, len(pod.cumulative_energy_) + 1)
    ax.plot(modes, pod.cumulative_energy_, "o-", ms=3, lw=1.5, color="steelblue")
    ax.axhline(0.90, color="red", ls="--", lw=1, label="90%")
    ax.axhline(0.99, color="orange", ls="--", lw=1, label="99%")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Cumulative energy")
    ax.set_title("Cumulative POD energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)

    if standalone:
        plt.tight_layout()
    return fig


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    ax: Optional[plt.Axes] = None,
    n_plot: int = 50_000,
) -> Figure:
    """Predicted-vs-actual scatter plot with 1:1 line.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape [n_samples]
    n_plot : int
        Maximum points to plot (random subsample for large arrays).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    if len(y_true) > n_plot:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_true), n_plot, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]

    ax.scatter(y_true, y_pred, s=1, alpha=0.3, color="steelblue", rasterized=True)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
    ax.set_xlabel("Actual heat flux (W/m²)")
    ax.set_ylabel("Predicted heat flux (W/m²)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
    return fig


def plot_model_summary(
    q_true_field: np.ndarray,
    q_pred_field: np.ndarray,
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    metrics_train,
    metrics_test,
    r2_t: Optional[np.ndarray] = None,
    rmse_t: Optional[np.ndarray] = None,
    timestep_idx: int = 0,
    model_name: str = "Model",
) -> Figure:
    """Six-panel summary plot matching the paper figure style.

    Panels: scatter | true field | predicted field | residual | RMSE(t) | R²(t)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{model_name} Evaluation", fontsize=13)

    # (a) Scatter
    plot_scatter(y_true_flat, y_pred_flat, title="Predicted vs Actual", ax=axes[0, 0])

    # (b) True field at timestep
    plot_field(
        q_true_field[:, :, timestep_idx],
        title=f"Actual (t={timestep_idx})",
        ax=axes[0, 1],
    )

    # (c) Predicted field
    plot_field(
        q_pred_field[:, :, timestep_idx],
        title=f"Predicted (t={timestep_idx})",
        ax=axes[0, 2],
    )

    # (d) Residual
    residual = q_true_field[:, :, timestep_idx] - q_pred_field[:, :, timestep_idx]
    vr = np.abs(residual).max()
    im = axes[1, 0].imshow(
        residual, cmap="RdBu_r", vmin=-vr, vmax=vr, origin="upper"
    )
    plt.colorbar(im, ax=axes[1, 0], label="Residual (W/m²)")
    axes[1, 0].set_title("Residual")

    # (e) Time-resolved RMSE
    if rmse_t is not None:
        axes[1, 1].plot(rmse_t, lw=0.9, color="steelblue")
        axes[1, 1].axhline(rmse_t.mean(), color="red", ls="--", lw=1,
                           label=f"Mean = {rmse_t.mean():,.0f}")
        axes[1, 1].set_xlabel("Test time step")
        axes[1, 1].set_ylabel("RMSE (W/m²)")
        axes[1, 1].set_title("Time-resolved RMSE")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # (f) Time-resolved R²
    if r2_t is not None:
        axes[1, 2].plot(r2_t, lw=0.9, color="darkorange")
        axes[1, 2].axhline(r2_t.mean(), color="red", ls="--", lw=1,
                           label=f"Mean R² = {r2_t.mean():.4f}")
        axes[1, 2].set_xlabel("Test time step")
        axes[1, 2].set_ylabel("R²")
        axes[1, 2].set_title("Time-resolved R²")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    # Metrics text
    fig.text(0.01, 0.01,
             f"Train: {metrics_train}    Test: {metrics_test}",
             fontsize=8, va="bottom")

    plt.tight_layout()
    return fig
