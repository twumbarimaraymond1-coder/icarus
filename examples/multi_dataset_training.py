"""
examples/multi_dataset_training.py
====================================
Shows the complete workflow for registering multiple datasets,
extracting POD modal features, and training a cross-dataset model.

Usage
-----
    python examples/multi_dataset_training.py
    python examples/multi_dataset_training.py --d001 /path/to/data.mat --d002 /path/to/data2.mat
"""

import argparse, os, tempfile
import numpy as np
import matplotlib; matplotlib.use("Agg")

from icarus.registry.dataset import DatasetRegistry, DatasetEntry
from icarus.registry.extractor import FeatureExtractor
from icarus.registry.trainer import MultiDatasetTrainer
from icarus.data.loader import from_arrays
from icarus.models.neural import HeatFluxNet


def make_synthetic(ny=30, nx=50, nt=200, T_mean=427.0, q_mean=300_000, seed=42):
    rng = np.random.default_rng(seed)
    XX, YY = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    t = np.linspace(0, 1, nt)
    m1 = np.sin(np.pi*YY)*np.cos(np.pi*XX)
    m2 = np.cos(2*np.pi*YY)*np.sin(2*np.pi*XX)
    T = T_mean + m1[:,:,None]*3*np.sin(2*np.pi*t)[None,None,:] + m2[:,:,None]*2*np.cos(4*np.pi*t)[None,None,:] + 0.5*rng.standard_normal((ny,nx,nt))
    q = q_mean - 12000*(T-T_mean) + 8000*rng.standard_normal((ny,nx,nt))
    return from_arrays(T, q, dt=2.5e-4)


def to_npz(data):
    p = tempfile.mktemp(suffix=".npz")
    np.savez(p, T=data["T"], qL2=data["q"], TimeStep=data["dt"])
    return p


def main(d001_path=None, d002_path=None):
    print("="*60)
    print("  Icarus multi-dataset training example")
    print("="*60)

    reg = DatasetRegistry("~/.icarus/datasets")
    tmp = []

    # D001
    if not d001_path:
        d001_path = to_npz(make_synthetic(seed=42, T_mean=427.0, q_mean=300_000))
        tmp.append(d001_path); print("\nD001: synthetic water flow boiling")
    reg.register(DatasetEntry("D001","water","flow_boiling","plain_copper","Loughborough",raw_path=d001_path,spatial_crop=2))

    # D002
    if not d002_path:
        d002_path = to_npz(make_synthetic(seed=99, T_mean=320.0, q_mean=250_000))
        tmp.append(d002_path); print("D002: synthetic FC-72 pool boiling")
    reg.register(DatasetEntry("D002","FC-72","pool_boiling","micropillar","Example Uni",raw_path=d002_path,spatial_crop=2))

    reg.summary()

    print("\n[3] Extracting features...")
    ext = FeatureExtractor(reg, n_pod_modes=5)
    for ds in ["D001","D002"]:
        ext.process(ds, temperature_key="T", heatflux_key="qL2", timestep_key="TimeStep", force=True)

    reg.summary()

    print("\n[4] Training cross-dataset model...")
    trainer = MultiDatasetTrainer(reg, n_pod_modes=5, n_training_samples=50_000, optimise_hyperparams=False)
    trainer.model_ = HeatFluxNet(strategy="modal", hidden_layer_sizes=(128,128), activation="tanh", alpha=1e-3, learning_rate_init=1e-3, max_iter=100)
    trainer.fit(dataset_ids=["D001","D002"], verbose=True)

    print("\n[5] Evaluating...")
    metrics = trainer.evaluate()

    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/multi_dataset_v0.1.pkl")

    saved = MultiDatasetTrainer.load_model("models/multi_dataset_v0.1.pkl")
    print(f"\nLoaded model — datasets: {saved['dataset_ids']}")
    print(f"Test R²: {metrics['test'].r2:.4f}  RMSE: {metrics['test'].rmse:,.0f} W/m²")

    for f in tmp:
        if os.path.exists(f): os.unlink(f)
    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d001", default=None)
    p.add_argument("--d002", default=None)
    a = p.parse_args()
    main(a.d001, a.d002)
