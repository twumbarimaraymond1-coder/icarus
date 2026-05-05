"""
icarus.metrics.evaluation
=============================
Standard error metrics for heat flux prediction evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@dataclass
class Metrics:
    """Container for model evaluation results.

    Attributes
    ----------
    r2 : float
    rmse : float   W/m²
    mae : float    W/m²
    split : str    "train" or "test"
    """
    r2: float
    rmse: float
    mae: float
    split: str = "test"

    def __str__(self) -> str:
        return (
            f"[{self.split}]  R² = {self.r2:.4f}  "
            f"RMSE = {self.rmse:,.0f} W/m²  "
            f"MAE = {self.mae:,.0f} W/m²"
        )


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str = "test",
) -> Metrics:
    """Compute R², RMSE, and MAE between true and predicted heat flux.

    Parameters
    ----------
    y_true : np.ndarray, shape [n_samples]
    y_pred : np.ndarray, shape [n_samples]
    split : str
        Label for the metrics container ("train" or "test").

    Returns
    -------
    Metrics

    Examples
    --------
    >>> m = evaluate(q_true, q_pred, split="test")
    >>> print(m)
    [test]  R² = 0.7293  RMSE = 25,959 W/m²  MAE = 20,656 W/m²
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return Metrics(r2=r2, rmse=rmse, mae=mae, split=split)


def evaluate_timeresolved(
    q_true: np.ndarray,
    q_pred: np.ndarray,
    ny: int,
    nx: int,
) -> dict[str, np.ndarray]:
    """Compute time-resolved R² and RMSE across spatial fields.

    Parameters
    ----------
    q_true : np.ndarray, shape [ny*nx, nt] or [ny, nx, nt]
    q_pred : np.ndarray, shape [ny*nx, nt] or [ny, nx, nt]
    ny, nx : int
        Spatial dimensions.

    Returns
    -------
    dict with keys ``"r2_t"`` and ``"rmse_t"``, each shape [nt].
    """
    if q_true.ndim == 3:
        q_true = q_true.reshape(ny * nx, -1)
        q_pred = q_pred.reshape(ny * nx, -1)

    nt = q_true.shape[1]
    r2_t = np.zeros(nt)
    rmse_t = np.zeros(nt)

    for t in range(nt):
        r2_t[t] = r2_score(q_true[:, t], q_pred[:, t])
        rmse_t[t] = np.sqrt(mean_squared_error(q_true[:, t], q_pred[:, t]))

    return {"r2_t": r2_t, "rmse_t": rmse_t}
