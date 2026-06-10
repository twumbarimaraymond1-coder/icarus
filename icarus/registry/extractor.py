"""
icarus.registry.extractor
==========================
Processes raw experimental datasets into POD modal features
and stores them in HDF5 format for efficient multi-dataset training.

Each processed dataset produces a ``features.h5`` file containing:

    /T_contribs     [n_samples, n_modes]  temperature modal contributions
    /q_contribs     [n_samples, n_modes]  heat flux modal contributions
    /q_flat         [n_samples]           raw heat flux, TIME-major (legacy)
    /q_abs          [n_samples]           absolute heat flux, TIME-major,
                                          row-aligned with q_contribs
    /q_mean_field   [n_pix]               per-pixel time-mean heat flux

All per-sample arrays are TIME-major (row = t * n_pix + p), so the split
mask's first ``nt_train * n_pix`` rows are a genuine temporal train window.
    /split          [n_samples]           0=train, 1=test (integer mask)
    /metadata       (HDF5 attributes)     fluid, setup, n_modes, etc.

The POD bases are saved separately as ``.npz`` files so they can be
reused for inference on new data without re-fitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import h5py

from icarus.data.loader import load
from icarus.data.preprocessor import Preprocessor, PreprocessorConfig
from icarus.decomposition.pod import POD
from icarus.features.engineer import flatten_target
from icarus.registry.dataset import DatasetRegistry


class FeatureExtractor:
    """Extract and store POD modal features from a raw dataset.

    Parameters
    ----------
    registry : DatasetRegistry
        The dataset registry managing storage paths.
    n_pod_modes : int
        Number of POD modes to extract (default 5).

    Examples
    --------
    >>> from icarus.registry.dataset import DatasetRegistry
    >>> from icarus.registry.extractor import FeatureExtractor
    >>>
    >>> reg = DatasetRegistry("~/.icarus/datasets")
    >>> extractor = FeatureExtractor(reg, n_pod_modes=5)
    >>> extractor.process("D001")
    >>> extractor.process("D002")
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        n_pod_modes: int = 5,
    ):
        self.registry = registry
        self.n_pod_modes = n_pod_modes

    def process(
        self,
        dataset_id: str,
        temperature_key: str = "T",
        heatflux_key: str = "qL2",
        timestep_key: str = "TimeStep",
        force: bool = False,
    ) -> Path:
        """Process a registered raw dataset into HDF5 modal features.

        Parameters
        ----------
        dataset_id : str
        temperature_key, heatflux_key, timestep_key : str
            Variable names in the source file.
        force : bool
            Re-process even if features already exist.

        Returns
        -------
        Path to the output ``features.h5`` file.
        """
        entry = self.registry.get(dataset_id)

        out_dir = self.registry.processed_dir / dataset_id
        out_dir.mkdir(exist_ok=True)
        features_path = out_dir / "features.h5"

        if features_path.exists() and not force:
            print(f"[extractor] {dataset_id} already processed. "
                  f"Use force=True to re-run.")
            return features_path

        print(f"[extractor] Processing {dataset_id} ({entry.fluid} "
              f"{entry.setup})...")

        # ── 1. Load ──────────────────────────────────────────────────────────
        print(f"  Loading from {entry.raw_path}...")
        data = load(
            entry.raw_path,
            temperature_key=temperature_key,
            heatflux_key=heatflux_key,
            timestep_key=timestep_key,
        )

        # ── 2. Preprocess ────────────────────────────────────────────────────
        cfg = PreprocessorConfig(
            spatial_crop=entry.spatial_crop,
            trim_frames=entry.trim_frames,
        )
        pre = Preprocessor(cfg)
        out = pre.fit_transform(data)
        T, q = out["T"], out["q"]
        T_c, q_c = out["T_c"], out["q_c"]
        ny, nx, nt = T.shape
        n_pix = ny * nx
        print(f"  Shape after preprocessing: {T.shape}")

        # ── 3. POD decomposition ─────────────────────────────────────────────
        print(f"  Fitting POD ({self.n_pod_modes} modes)...")
        X_c_T = Preprocessor.to_matrix(T_c)
        X_c_q = Preprocessor.to_matrix(q_c)

        nt_train = int(nt * entry.train_fraction)
        pod_T = POD(n_modes=self.n_pod_modes).fit(X_c_T[:, :nt_train])
        pod_q = POD(n_modes=self.n_pod_modes).fit(X_c_q[:, :nt_train])

        cum_T = pod_T.cumulative_energy_[self.n_pod_modes - 1]
        cum_q = pod_q.cumulative_energy_[self.n_pod_modes - 1]
        print(f"  T energy ({self.n_pod_modes} modes): {cum_T:.1%}")
        print(f"  q energy ({self.n_pod_modes} modes): {cum_q:.1%}")

        # ── 4. Modal contributions ───────────────────────────────────────────
        print("  Computing modal contributions...")
        # Chunk over time + emit float32 to bound peak memory on large real
        # datasets (millions of pixel-timesteps). Downstream casts to float32
        # anyway, so single precision here is lossless for the pipeline.
        T_contribs = pod_T.modal_contributions(
            X_c_T, chunk_size=512, dtype=np.float32)  # [n_pix, nt, modes]
        q_contribs = pod_q.modal_contributions(
            X_c_q, chunk_size=512, dtype=np.float32)

        # Reshape to [n_samples, n_modes], TIME-major (row = t*n_pix + p).
        # The split mask below marks the first nt_train*n_pix rows as train,
        # which is only a temporal split under time-major ordering — a
        # pixel-major reshape here would silently turn it into a spatial pixel
        # split (train/test sharing every timestep). Guarded by
        # tests/test_core.py::TestExtractorTimeMajorOrdering.
        X_feat = (T_contribs.transpose(1, 0, 2)
                  .reshape(n_pix * nt, self.n_pod_modes).astype(np.float32))
        y_feat = (q_contribs.transpose(1, 0, 2)
                  .reshape(n_pix * nt, self.n_pod_modes).astype(np.float32))
        q_flat = flatten_target(q)                      # [n_pix * nt], time-major

        # Absolute (un-centred) heat flux + per-pixel mean field, time-major so
        # they are row-aligned with q_contribs/T_contribs. These let the trainer
        # report a true absolute-field R² alongside the fluctuation R².
        q_abs = flatten_target(q).astype(np.float32)             # [n_pix*nt]
        q_mean_field = q.mean(axis=2).reshape(n_pix).astype(np.float32)  # [n_pix]

        # Train/test mask (0=train, 1=test)
        split_mask = np.ones(n_pix * nt, dtype=np.int8)
        split_mask[:nt_train * n_pix] = 0

        # ── 5. Save features to HDF5 ─────────────────────────────────────────
        print(f"  Saving features to {features_path}...")
        with h5py.File(features_path, "w") as f:
            f.create_dataset("T_contribs", data=X_feat,
                             compression="gzip", compression_opts=4)
            f.create_dataset("q_contribs", data=y_feat,
                             compression="gzip", compression_opts=4)
            f.create_dataset("q_flat", data=q_flat,
                             compression="gzip", compression_opts=4)
            f.create_dataset("q_abs", data=q_abs,
                             compression="gzip", compression_opts=4)
            f.create_dataset("q_mean_field", data=q_mean_field,
                             compression="gzip", compression_opts=4)
            f.create_dataset("split", data=split_mask)

            # Store metadata as HDF5 attributes
            f.attrs["dataset_id"] = dataset_id
            f.attrs["fluid"] = entry.fluid
            f.attrs["setup"] = entry.setup
            f.attrs["surface"] = entry.surface
            f.attrs["n_pod_modes"] = self.n_pod_modes
            f.attrs["n_samples"] = n_pix * nt
            f.attrs["ny"] = ny
            f.attrs["nx"] = nx
            f.attrs["nt"] = nt
            f.attrs["nt_train"] = nt_train
            f.attrs["cum_energy_T"] = float(cum_T)
            f.attrs["cum_energy_q"] = float(cum_q)
            f.attrs["q_mean"] = float(q.mean())
            f.attrs["q_std"] = float(q.std())

        # ── 6. Save POD bases ────────────────────────────────────────────────
        pod_T_path = out_dir / "pod_T.npz"
        pod_q_path = out_dir / "pod_q.npz"

        np.savez_compressed(
            pod_T_path,
            modes=pod_T.modes_,
            singular_values=pod_T.singular_values_,
            energy_fractions=pod_T.energy_fractions_,
            T_mean=pre.T_mean,
        )
        np.savez_compressed(
            pod_q_path,
            modes=pod_q.modes_,
            singular_values=pod_q.singular_values_,
            energy_fractions=pod_q.energy_fractions_,
            q_mean=pre.q_mean,
        )

        # ── 7. Update registry ───────────────────────────────────────────────
        self.registry._index[dataset_id]["processed_path"] = str(features_path)
        self.registry._index[dataset_id]["status"] = "features_extracted"
        self.registry._index[dataset_id]["n_samples"] = int(n_pix * nt)
        self.registry._save_index()

        print(f"  Done. {n_pix * nt:,} samples extracted.")
        return features_path
