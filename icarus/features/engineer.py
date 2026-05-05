"""
icarus.features.engineer
============================
Feature construction for heat flux prediction models.

Three feature strategies mirror the three models in the paper:

Model A  – raw temperature only (no feature engineering needed)
Model B  – temperature + spatial and temporal gradients
Model C  – POD modal contributions (recommended: best performance)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def build_gradient_features(
    T: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Build spatiotemporal gradient features from a temperature field.

    Computes dT/dt, dT/dx, dT/dy and stacks them with the raw temperature
    to form the Model B feature set.

    Parameters
    ----------
    T : np.ndarray, shape [ny, nx, nt]
        Temperature field (raw, not mean-subtracted).
    dt : float
        Timestep in seconds for the temporal gradient.

    Returns
    -------
    np.ndarray, shape [ny*nx*nt, 4]
        Columns: [T, dT/dt, dT/dx, dT/dy].

    Examples
    --------
    >>> X_B = build_gradient_features(T, dt=2.5e-4)
    >>> X_B.shape   # (ny*nx*nt, 4)
    """
    dTdt = np.gradient(T, dt, axis=2)
    dTdx = np.gradient(T, axis=1)
    dTdy = np.gradient(T, axis=0)

    ny, nx, nt = T.shape
    n = ny * nx * nt

    features = np.column_stack([
        T.reshape(n),
        dTdt.reshape(n),
        dTdx.reshape(n),
        dTdy.reshape(n),
    ])
    return features.astype(np.float32)


def build_modal_features(
    T_contribs: np.ndarray,
    q_contribs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Build Model C features from POD modal contribution fields.

    Flattens the [n_pix, nt, n_modes] contribution arrays into
    [n_pix*nt, n_modes] feature matrices suitable for sklearn estimators.

    Parameters
    ----------
    T_contribs : np.ndarray, shape [n_pix, nt, n_modes]
        Temperature modal contributions from :meth:`POD.modal_contributions`.
    q_contribs : np.ndarray, shape [n_pix, nt, n_modes], optional
        Heat flux modal contributions (targets for training).

    Returns
    -------
    X : np.ndarray, shape [n_pix*nt, n_modes]
        Feature matrix (temperature modal contributions).
    y : np.ndarray, shape [n_pix*nt, n_modes] or None
        Target matrix (heat flux modal contributions), if provided.

    Examples
    --------
    >>> pod_T = POD(n_modes=5).fit(X_c_T)
    >>> pod_q = POD(n_modes=5).fit(X_c_q)
    >>> T_contribs = pod_T.modal_contributions(X_c_T)
    >>> q_contribs = pod_q.modal_contributions(X_c_q)
    >>> X, y = build_modal_features(T_contribs, q_contribs)
    """
    n_pix, nt, n_modes = T_contribs.shape

    X = T_contribs.reshape(n_pix * nt, n_modes).astype(np.float32)
    y = None
    if q_contribs is not None:
        y = q_contribs.reshape(n_pix * nt, n_modes).astype(np.float32)

    return X, y


def build_raw_features(T: np.ndarray) -> np.ndarray:
    """Flatten raw temperature into a 2-D feature matrix (Model A).

    Parameters
    ----------
    T : np.ndarray, shape [ny, nx, nt]

    Returns
    -------
    np.ndarray, shape [ny*nx*nt, 1]
    """
    return T.reshape(-1, 1).astype(np.float32)


def flatten_target(q: np.ndarray) -> np.ndarray:
    """Flatten a heat flux field into a 1-D target vector.

    Parameters
    ----------
    q : np.ndarray, shape [ny, nx, nt]

    Returns
    -------
    np.ndarray, shape [ny*nx*nt]
    """
    return q.reshape(-1).astype(np.float32)


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.7,
    n_pix: Optional[int] = None,
    nt: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and targets along the time axis.

    Respects the temporal ordering of the data — the test set corresponds
    to later timesteps, not a random shuffle. This mirrors the approach in
    the paper and avoids data leakage.

    Parameters
    ----------
    X : np.ndarray, shape [n_samples, n_features]
    y : np.ndarray, shape [n_samples] or [n_samples, n_targets]
    train_fraction : float
        Fraction of *timesteps* to use for training.
    n_pix : int, optional
        Number of pixels per frame. Required if X was built from a
        [n_pix, nt, ...] array so the split can be done on timesteps.
        If None, a simple index split is used.
    nt : int, optional
        Total number of timesteps. Required alongside n_pix.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if n_pix is not None and nt is not None:
        # Split on timestep boundaries
        nt_train = int(nt * train_fraction)
        train_idx = np.arange(nt_train * n_pix)
        test_idx = np.arange(nt_train * n_pix, nt * n_pix)
    else:
        split = int(len(X) * train_fraction)
        train_idx = np.arange(split)
        test_idx = np.arange(split, len(X))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
