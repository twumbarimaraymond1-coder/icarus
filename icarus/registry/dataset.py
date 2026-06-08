"""
icarus.registry.dataset
========================
Manages the Icarus dataset registry — metadata, storage locations,
and retrieval of contributed experimental datasets.

Each dataset entry records:
- Source metadata (fluid, surface, setup, contributor)
- Storage location (local path or cloud URI)
- Processing status (raw / features_extracted / validated)
- POD configuration used during feature extraction

Datasets are stored as HDF5 files after processing, with a JSON
registry index tracking all available datasets.

Storage layout
--------------
<registry_root>/
    registry.json               ← index of all datasets
    raw/
        D001_water_flow/
            data.mat            ← original contributor file
            metadata.json       ← contributor-provided metadata
    processed/
        D001_water_flow/
            features.h5         ← extracted POD modal features
            pod_T.npz           ← fitted temperature POD basis
            pod_q.npz           ← fitted heat flux POD basis
            stats.json          ← dataset statistics
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal


Status = Literal["raw", "features_extracted", "validated"]


@dataclass
class DatasetEntry:
    """Metadata record for one contributed dataset.

    Attributes
    ----------
    dataset_id : str
        Unique identifier e.g. ``"D001"``.
    fluid : str
        Working fluid e.g. ``"water"``, ``"FC-72"``, ``"R134a"``.
    setup : str
        Experimental setup e.g. ``"flow_boiling"``, ``"pool_boiling"``.
    surface : str
        Heater surface description.
    contributor : str
        Institution or author name.
    n_samples : int
        Total pixel-time samples after preprocessing.
    status : str
        Processing status: ``"raw"``, ``"features_extracted"``,
        ``"validated"``.
    raw_path : str
        Path to the original data file (local or cloud URI).
    processed_path : str or None
        Path to the processed HDF5 features file.
    doi : str or None
        Published DOI if available.
    notes : str
        Free-text notes.
    """
    dataset_id: str
    fluid: str
    setup: str
    surface: str
    contributor: str
    n_samples: int = 0
    status: Status = "raw"
    raw_path: str = ""
    processed_path: Optional[str] = None
    doi: Optional[str] = None
    notes: str = ""
    spatial_crop: int = 5
    trim_frames: int = 0
    n_pod_modes: int = 5
    train_fraction: float = 0.7


class DatasetRegistry:
    """Registry of all Icarus datasets.

    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.
        Created automatically if it does not exist.

    Examples
    --------
    >>> from icarus.registry.dataset import DatasetRegistry, DatasetEntry
    >>> reg = DatasetRegistry("~/.icarus/datasets")
    >>> entry = DatasetEntry(
    ...     dataset_id="D001",
    ...     fluid="water",
    ...     setup="flow_boiling",
    ...     surface="plain_copper",
    ...     contributor="Loughborough University",
    ...     raw_path="/path/to/data.mat",
    ... )
    >>> reg.register(entry)
    >>> reg.list_datasets()
    """

    def __init__(self, root: str | Path = "~/.icarus/datasets"):
        self.root = Path(root).expanduser().resolve()
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self._index_path = self.root / "registry.json"

        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        self._index: dict[str, dict] = self._load_index()

    # ── public API ────────────────────────────────────────────────────────────

    def register(self, entry: DatasetEntry) -> None:
        """Add or update a dataset entry in the registry.

        Parameters
        ----------
        entry : DatasetEntry
        """
        self._index[entry.dataset_id] = asdict(entry)
        self._save_index()
        print(f"[registry] Registered {entry.dataset_id}: "
              f"{entry.fluid} {entry.setup} ({entry.contributor})")

    def get(self, dataset_id: str) -> DatasetEntry:
        """Retrieve a dataset entry by ID.

        Parameters
        ----------
        dataset_id : str

        Returns
        -------
        DatasetEntry

        Raises
        ------
        KeyError if dataset_id not found.
        """
        if dataset_id not in self._index:
            raise KeyError(
                f"Dataset '{dataset_id}' not found. "
                f"Available: {list(self._index.keys())}"
            )
        return DatasetEntry(**self._index[dataset_id])

    def list_datasets(self, status: Optional[Status] = None) -> list[DatasetEntry]:
        """List all registered datasets, optionally filtered by status.

        Parameters
        ----------
        status : str, optional
            Filter by ``"raw"``, ``"features_extracted"``, or ``"validated"``.

        Returns
        -------
        list[DatasetEntry]
        """
        entries = [DatasetEntry(**v) for v in self._index.values()]
        if status:
            entries = [e for e in entries if e.status == status]
        return entries

    def import_file(
        self,
        dataset_id: str,
        source_path: str | Path,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Copy a raw data file into the registry storage.

        Parameters
        ----------
        dataset_id : str
        source_path : str or Path
            Path to the contributor's data file.
        metadata : dict, optional
            Contributor metadata to save alongside the data.

        Returns
        -------
        Path : destination path inside the registry.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        dest_dir = self.raw_dir / dataset_id
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / source.name
        shutil.copy2(source, dest)

        if metadata:
            meta_path = dest_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Update registry entry if it exists
        if dataset_id in self._index:
            self._index[dataset_id]["raw_path"] = str(dest)
            self._save_index()

        print(f"[registry] Imported {source.name} → {dest}")
        return dest

    def summary(self) -> None:
        """Print a formatted summary of all registered datasets."""
        entries = self.list_datasets()
        if not entries:
            print("No datasets registered.")
            return

        print(f"\n{'ID':<8} {'Fluid':<12} {'Setup':<20} "
              f"{'Samples':>12} {'Status':<20} {'Contributor'}")
        print("-" * 85)
        total = 0
        for e in entries:
            print(f"{e.dataset_id:<8} {e.fluid:<12} {e.setup:<20} "
                  f"{e.n_samples:>12,} {e.status:<20} {e.contributor}")
            total += e.n_samples
        print("-" * 85)
        print(f"{'Total':<41} {total:>12,}")

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)
