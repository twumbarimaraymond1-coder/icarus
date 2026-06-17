"""
icarus.data.loader
=====================
Unified loader for IR thermography datasets stored in various formats.

Supported formats
-----------------
- NumPy   : .npy, .npz
- MATLAB  : .mat  (v5 and v7.3 / HDF5-based)
- HDF5    : .h5, .hdf5
- Raw     : pass numpy arrays directly

All loaders return arrays with shape convention [ny, nx, nt] for
three-dimensional fields (temperature / heat flux at the heater surface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def load(
    path: Union[str, Path],
    temperature_key: str = "T",
    heatflux_key: str = "qL2",
    timestep_key: str = "TimeStep",
) -> dict[str, np.ndarray]:
    """Load a thermography dataset from disk.

    Parameters
    ----------
    path : str or Path
        Path to the data file (.npy, .npz, .mat, .h5, .hdf5).
    temperature_key : str
        Key / variable name for the temperature array.
    heatflux_key : str
        Key / variable name for the heat flux array.
    timestep_key : str
        Key / variable name for the timestep array or scalar.

    Returns
    -------
    dict with keys ``"T"``, ``"q"``, ``"dt"`` (float, seconds).
    T and q have shape [ny, nx, nt] after loading.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file format is not supported or required keys are missing.

    Examples
    --------
    >>> data = load("experiment.mat", temperature_key="T", heatflux_key="qL2")
    >>> T, q, dt = data["T"], data["q"], data["dt"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return _load_npy(path)
    elif suffix == ".npz":
        return _load_npz(path, temperature_key, heatflux_key, timestep_key)
    elif suffix == ".mat":
        return _load_mat(path, temperature_key, heatflux_key, timestep_key)
    elif suffix in {".h5", ".hdf5"}:
        return _load_hdf5(path, temperature_key, heatflux_key, timestep_key)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Supported: .npy, .npz, .mat, .h5, .hdf5"
        )


def load_field(
    path: Union[str, Path],
    field: str = "heatflux",
    *,
    temperature_key: str = "T",
    heatflux_key: str = "qL2",
    timestep_key: str = "TimeStep",
) -> tuple[np.ndarray, float]:
    """Load a single field as ``[ny, nx, nt]`` plus ``dt``, reading minimally.

    Unlike :func:`load` (which reads every variable), this reads only the one
    field you ask for. For a 4-D temperature array in a v7.3 ``.mat`` file it
    slices only the heater surface layer (z index 0) off disk, so memory use
    stays low (~one layer instead of the full volume — important for the large
    real datasets).

    Parameters
    ----------
    path : str or Path
    field : {"heatflux", "temperature"}
        Which field to read. "temp" is accepted as an alias for temperature.
    temperature_key, heatflux_key, timestep_key : str
        Variable names in the source file.

    Returns
    -------
    (field_array, dt) : (np.ndarray [ny, nx, nt], float)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    key = temperature_key if field in ("temp", "temperature") else heatflux_key
    suffix = path.suffix.lower()

    if suffix in {".mat", ".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            if key not in f:
                raise ValueError(
                    f"'{key}' not found in {path}. Available: {list(f.keys())}"
                )
            dataset = f[key]
            # MATLAB stores arrays axis-reversed.
            if dataset.ndim == 4:
                # stored [nt, nz, nx, ny] -> heater layer 0 -> [nt, nx, ny]
                raw = np.asarray(dataset[:, 0, :, :], dtype=np.float64)
            elif dataset.ndim == 3:
                raw = np.asarray(dataset[...], dtype=np.float64)  # [nt, nx, ny]
            else:
                raise ValueError(
                    f"Expected 3-D or 4-D '{key}', got ndim={dataset.ndim}."
                )
            field_array = raw.transpose(2, 1, 0)        # -> [ny, nx, nt]
            if timestep_key in f:
                dt = float(np.array(f[timestep_key]).flat[0])
            else:
                dt = 1.0
        return field_array, dt

    # .npz / .npy
    archive = np.load(path)
    if key not in archive:
        raise ValueError(f"'{key}' not found in {path}.")
    field_array = np.asarray(archive[key], dtype=np.float64)
    if field_array.ndim == 4:
        field_array = field_array[:, :, 0, :]
    dt = float(archive[timestep_key]) if timestep_key in archive else 1.0
    return field_array, dt


def from_arrays(
    T: np.ndarray,
    q: np.ndarray,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    """Create a dataset dict directly from numpy arrays.

    Parameters
    ----------
    T : np.ndarray
        Temperature field, shape [ny, nx, nt] or [ny, nx, nz, nt].
        If 4-D, the first z-layer (heater surface) is extracted automatically.
    q : np.ndarray
        Heat flux field, shape [ny, nx, nt].
    dt : float
        Timestep in seconds.

    Returns
    -------
    dict with keys ``"T"``, ``"q"``, ``"dt"``.
    """
    T = np.asarray(T, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if T.ndim == 4:
        T = T[:, :, 0, :]  # extract heater surface layer

    _validate_shapes(T, q)
    return {"T": T, "q": q, "dt": float(dt)}


# ─── private helpers ──────────────────────────────────────────────────────────

def _validate_shapes(T: np.ndarray, q: np.ndarray) -> None:
    if T.ndim != 3:
        raise ValueError(f"Temperature array must be 3-D [ny, nx, nt], got {T.shape}")
    if q.ndim != 3:
        raise ValueError(f"Heat flux array must be 3-D [ny, nx, nt], got {q.shape}")
    if T.shape != q.shape:
        raise ValueError(
            f"Temperature {T.shape} and heat flux {q.shape} shapes must match."
        )


def _load_npy(path: Path) -> dict[str, np.ndarray]:
    raise ValueError(
        "Single .npy files can only hold one array. "
        "Use from_arrays() or save as .npz with keys 'T', 'q', 'dt'."
    )


def _load_npz(
    path: Path,
    T_key: str,
    q_key: str,
    dt_key: str,
) -> dict[str, np.ndarray]:
    archive = np.load(path)
    _check_keys(archive, [T_key, q_key], path)
    T = archive[T_key].astype(np.float64)
    q = archive[q_key].astype(np.float64)
    dt = float(archive[dt_key]) if dt_key in archive else 1.0
    if T.ndim == 4:
        T = T[:, :, 0, :]
    _validate_shapes(T, q)
    return {"T": T, "q": q, "dt": dt}


def _load_mat(
    path: Path,
    T_key: str,
    q_key: str,
    dt_key: str,
) -> dict[str, np.ndarray]:
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(path), squeeze_me=False)
        if T_key in mat:
            # v5 .mat file loaded successfully
            _check_keys(mat, [T_key, q_key], path)
            T = np.asarray(mat[T_key], dtype=np.float64)
            q = np.asarray(mat[q_key], dtype=np.float64)
            dt_raw = mat.get(dt_key, np.array([[1.0]]))
            dt = float(np.squeeze(dt_raw).flat[0]) if np.squeeze(dt_raw).ndim >= 1 else float(dt_raw)
            if T.ndim == 4:
                T = T[:, :, 0, :]
            _validate_shapes(T, q)
            return {"T": T, "q": q, "dt": dt}
    except Exception:
        pass

    # Fall back to HDF5 for v7.3 .mat files
    return _load_hdf5(path, T_key, q_key, dt_key)


def _load_hdf5(
    path: Path,
    T_key: str,
    q_key: str,
    dt_key: str,
) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to load .h5 / .hdf5 / v7.3 .mat files.")

    with h5py.File(path, "r") as f:
        _check_keys(f, [T_key, q_key], path)
        T = np.array(f[T_key], dtype=np.float64)
        q = np.array(f[q_key], dtype=np.float64)
        dt = float(np.array(f[dt_key]).flat[0]) if dt_key in f else 1.0

    # HDF5-backed MATLAB files store arrays transposed
    if T.ndim == 4:
        T = T.transpose(3, 2, 1, 0)[:, :, 0, :]
    elif T.ndim == 3:
        T = T.transpose(2, 1, 0)

    if q.ndim == 3:
        q = q.transpose(2, 1, 0)

    _validate_shapes(T, q)
    return {"T": T, "q": q, "dt": dt}


def _check_keys(container, keys: list[str], path: Path) -> None:
    missing = [k for k in keys if k not in container]
    if missing:
        raise ValueError(
            f"Keys {missing} not found in {path}. "
            f"Available keys: {list(container.keys() if hasattr(container, 'keys') else [])}"
        )
