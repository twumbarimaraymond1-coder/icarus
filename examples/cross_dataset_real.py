"""
examples/cross_dataset_real.py
==============================
Cross-dataset generalisation on three real flow-boiling surfaces stored as
MATLAB v7.3 (.mat) files with variables T / qL2 / TimeStep. Trains on two
surfaces, holds out the third entirely, and writes a timestamped metrics
report + saved model.

Unlike cross_dataset_eval.py (synthetic), this expects real .mat files and
takes their folder via --data-dir, so it is portable across machines
(laptop / DigiLab / HPC).

Usage
-----
    python examples/cross_dataset_real.py --data-dir /path/to/folder
    python examples/cross_dataset_real.py --data-dir ./data --test D002
    python examples/cross_dataset_real.py --data-dir ./data --out ~/icarus_results

Expects, by default, files named:  MODEL_~1.MAT  MODEL_~2.MAT  MODEL_~3.MAT
(override with --files if yours differ). Each ~2.4 GB needs ~2-3 GB RAM to
load, processed one at a time.
"""

from __future__ import annotations
import argparse, json, datetime
from pathlib import Path

from icarus.registry.dataset import DatasetRegistry, DatasetEntry
from icarus.registry.extractor import FeatureExtractor
from icarus.registry.trainer import MultiDatasetTrainer

# Default file -> (dataset_id, surface label)
DEFAULT_FILES = ["MODEL_~1.MAT", "MODEL_~2.MAT", "MODEL_~3.MAT"]
IDS = ["D001", "D002", "D003"]

# Preprocessing / model config (per project notes)
N_POD_MODES        = 5
SPATIAL_CROP       = 5
TRIM_FRAMES        = 43
N_TRAINING_SAMPLES = 500_000
N_TRIALS           = 15
Z_LAYER            = 0   # heater-surface z index (the loader takes layer 0)


def main(data_dir: Path, test_id: str, files: list[str], out_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir not found: {data_dir}")
    surfaces = dict(zip(IDS, files))
    train_ids = [d for d in IDS if d != test_id]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir : {data_dir}")
    print(f"Train    : {train_ids}  |  Hold out: {test_id}")

    reg = DatasetRegistry(str(out_dir / "registry_store"))
    for ds_id, fname in surfaces.items():
        raw = data_dir / fname
        if not raw.exists():
            raise FileNotFoundError(
                f"{raw} not found. Pass --files if your names differ.")
        reg.register(DatasetEntry(
            ds_id, "water", "flow_boiling", f"patch_{ds_id[-1]}", "Loughborough",
            raw_path=str(raw), spatial_crop=SPATIAL_CROP,
            trim_frames=TRIM_FRAMES, n_pod_modes=N_POD_MODES,
        ))

    ext = FeatureExtractor(reg, n_pod_modes=N_POD_MODES)
    for ds_id in IDS:
        ext.process(ds_id, temperature_key="T", heatflux_key="qL2",
                    timestep_key="TimeStep")
    reg.summary()

    trainer = MultiDatasetTrainer(
        reg, n_pod_modes=N_POD_MODES, n_training_samples=N_TRAINING_SAMPLES,
        optimise_hyperparams=True, n_trials=N_TRIALS,
    )
    trainer.cross_dataset_fit(train_ids=train_ids, test_id=test_id)
    metrics = trainer.evaluate()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"cross_{'_'.join(train_ids)}__test_{test_id}_{stamp}"
    trainer.save_model(str(out_dir / f"{tag}.pkl"))

    report = {
        "train_ids": train_ids, "test_id": test_id, "timestamp": stamp,
        "n_pod_modes": N_POD_MODES, "spatial_crop": SPATIAL_CROP,
        "trim_frames": TRIM_FRAMES, "z_layer": Z_LAYER,
        "train": {"r2": metrics["train"].r2, "rmse": metrics["train"].rmse,
                  "mae": metrics["train"].mae},
        # "test" == fluctuation field (the scientifically meaningful metric)
        "test":  {"r2": metrics["test"].r2,  "rmse": metrics["test"].rmse,
                  "mae": metrics["test"].mae},
        "test_metric": "fluctuation (mean-subtracted) field",
    }
    if "test_absolute" in metrics:
        m_abs = metrics["test_absolute"]
        report["test_absolute"] = {"r2": m_abs.r2, "rmse": m_abs.rmse,
                                   "mae": m_abs.mae}
    (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"\nHeld-out {test_id} (fluctuation field):  R2={metrics['test'].r2:.4f}  "
          f"RMSE={metrics['test'].rmse:,.0f} W/m2")
    if "test_absolute" in metrics:
        print(f"Held-out {test_id} (absolute field):     "
              f"R2={metrics['test_absolute'].r2:.4f}  "
              f"RMSE={metrics['test_absolute'].rmse:,.0f} W/m2")
    print(f"Report -> {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Folder containing the three .mat files")
    p.add_argument("--test", default="D003", choices=IDS,
                   help="Dataset held out for testing (default D003)")
    p.add_argument("--files", nargs=3, default=DEFAULT_FILES,
                   help="Three .mat filenames in D001 D002 D003 order")
    p.add_argument("--out", default="icarus_results",
                   help="Output folder for model + report")
    a = p.parse_args()
    main(Path(a.data_dir), a.test, a.files, Path(a.out).expanduser())
