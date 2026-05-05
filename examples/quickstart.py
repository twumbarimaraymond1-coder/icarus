"""
examples/quickstart.py
======================
Demonstrates the full icarus pipeline on a synthetic flow boiling dataset.
Run this to verify your installation and see the library in action.

Usage:
    python examples/quickstart.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering for CI / script use
import matplotlib.pyplot as plt

import icarus as tf
from icarus.data.loader import from_arrays
from icarus.data.preprocessor import Preprocessor, PreprocessorConfig
from icarus.decomposition.pod import POD
from icarus.decomposition.dmd import DMD
from icarus.features.engineer import (
    build_raw_features, build_gradient_features,
    build_modal_features, flatten_target,
    train_test_split_temporal,
)
from icarus.models.neural import HeatFluxNet
from icarus.metrics.evaluation import evaluate, evaluate_timeresolved
from icarus.visualisation.plots import (
    plot_cumulative_energy, plot_pod_modes, plot_scatter,
)


def make_synthetic_dataset(ny=41, nx=106, nt=400, seed=42):
    """Generate a synthetic flow boiling dataset with realistic structure."""
    rng = np.random.default_rng(seed)

    # Base temperature field with spatial structure
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 1, nt)
    XX, YY = np.meshgrid(x, y)   # [ny, nx]

    # Mean temperature ~427 K
    T_mean = 427.0

    # Two dominant spatial modes (mimicking POD modes 1 and 2)
    mode1 = np.sin(np.pi * XX) * np.cos(np.pi * YY)
    mode2 = np.cos(2 * np.pi * XX) * np.sin(2 * np.pi * YY)

    # Temporal evolution of each mode
    a1 = 3.0 * np.sin(2 * np.pi * t)
    a2 = 2.0 * np.cos(4 * np.pi * t)

    # Build T with shape [ny, nx, nt] directly
    mode1 = np.sin(np.pi * YY) * np.cos(np.pi * XX)   # [ny, nx]
    mode2 = np.cos(2 * np.pi * YY) * np.sin(2 * np.pi * XX)
    T = T_mean + (
        mode1[:, :, np.newaxis] * a1[np.newaxis, np.newaxis, :]
        + mode2[:, :, np.newaxis] * a2[np.newaxis, np.newaxis, :]
        + 0.5 * rng.standard_normal((ny, nx, nt))
    )

    # Heat flux inversely related to temperature with nonlinear component
    q = (
        300_000
        - 12_000 * (T - T_mean)
        + 1_500 * (T - T_mean) ** 2
        + 8_000 * rng.standard_normal((ny, nx, nt))
    )

    return from_arrays(T, q, dt=2.5e-4)


def main():
    print("=" * 60)
    print("  icarus quickstart example")
    print("=" * 60)

    # ── 1. Create synthetic dataset ───────────────────────────────────────────
    print("\n[1] Generating synthetic dataset...")
    data = make_synthetic_dataset(ny=41, nx=106, nt=400)
    T_raw, q_raw = data["T"], data["q"]
    print(f"    T shape: {T_raw.shape}  |  q shape: {q_raw.shape}")

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    print("\n[2] Preprocessing...")
    cfg = PreprocessorConfig(spatial_crop=2, trim_frames=0)
    pre = Preprocessor(cfg)
    out = pre.fit_transform(data)
    T, q = out["T"], out["q"]
    T_c, q_c = out["T_c"], out["q_c"]
    ny, nx, nt = T.shape
    print(f"    After crop: {T.shape}")

    # ── 3. POD analysis ───────────────────────────────────────────────────────
    print("\n[3] Running POD...")
    X_c_T = Preprocessor.to_matrix(T_c)
    X_c_q = Preprocessor.to_matrix(q_c)

    pod_T = POD(n_modes=10).fit(X_c_T)
    pod_q = POD(n_modes=10).fit(X_c_q)

    print(f"    Temperature: first 5 modes = {pod_T.cumulative_energy_[4]:.1%} energy")
    print(f"    Heat flux:   first 5 modes = {pod_q.cumulative_energy_[4]:.1%} energy")

    # Cross-correlation between temperature and heat flux temporal coefficients
    print("\n    Temperature ↔ heat flux modal correlations:")
    for i in range(5):
        r = np.corrcoef(
            pod_T.temporal_coefficients_[i],
            pod_q.temporal_coefficients_[i],
        )[0, 1]
        print(f"      Mode {i+1}: r = {r:.3f}")

    # ── 4. DMD forecasting ────────────────────────────────────────────────────
    print("\n[4] Running DMD (heat flux forecasting)...")
    nt_train_dmd = int(0.7 * nt)
    q_mean_field = out["q_mean"].squeeze()
    q_mean_vec = q_mean_field.reshape(-1)

    dmd = DMD(energy_threshold=0.99, dt=data["dt"])
    dmd.fit(X_c_q[:, :nt_train_dmd])

    x_init = X_c_q[:, nt_train_dmd - 1]
    n_forecast = nt - nt_train_dmd
    X_forecast = dmd.forecast_from(x_init, n_steps=n_forecast, x_mean=q_mean_vec)

    q_test_true = X_c_q[:, nt_train_dmd:] + q_mean_vec[:, None]
    from sklearn.metrics import r2_score
    r2_dmd = r2_score(q_test_true.ravel(), X_forecast.ravel())
    print(f"    DMD test R²: {r2_dmd:.4f}")

    # ── 5. Model A — raw temperature ──────────────────────────────────────────
    print("\n[5] Training Model A (raw temperature)...")
    X_A = build_raw_features(T)
    y_flat = flatten_target(q)
    X_A_tr, X_A_te, y_tr, y_te = train_test_split_temporal(
        X_A, y_flat, train_fraction=0.7, n_pix=ny * nx, nt=nt
    )
    model_A = HeatFluxNet(
        strategy="raw",
        hidden_layer_sizes=(32, 32),
        activation="relu",
        alpha=5e-3,
        learning_rate_init=1e-3,
        max_iter=100,
    )
    model_A.fit(X_A_tr, y_tr, n_samples=50_000)
    m_A = evaluate(y_te, model_A.predict(X_A_te), split="test")
    print(f"    {m_A}")

    # ── 6. Model B — gradients ────────────────────────────────────────────────
    print("\n[6] Training Model B (temperature + gradients)...")
    X_B = build_gradient_features(T, dt=data["dt"])
    X_B_tr, X_B_te, _, _ = train_test_split_temporal(
        X_B, y_flat, train_fraction=0.7, n_pix=ny * nx, nt=nt
    )
    model_B = HeatFluxNet(
        strategy="gradient",
        hidden_layer_sizes=(64, 64, 64),
        activation="tanh",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=100,
    )
    model_B.fit(X_B_tr, y_tr, n_samples=50_000)
    m_B = evaluate(y_te, model_B.predict(X_B_te), split="test")
    print(f"    {m_B}")

    # ── 7. Model C — POD modal contributions ──────────────────────────────────
    print("\n[7] Training Model C (POD modal contributions)...")
    n_modes = 5
    pod_T5 = POD(n_modes=n_modes).fit(X_c_T)
    pod_q5 = POD(n_modes=n_modes).fit(X_c_q)

    T_contribs = pod_T5.modal_contributions(X_c_T)
    q_contribs = pod_q5.modal_contributions(X_c_q)

    X_C, y_C = build_modal_features(T_contribs, q_contribs)
    X_C_tr, X_C_te, y_C_tr, y_C_te = train_test_split_temporal(
        X_C, y_C, train_fraction=0.7, n_pix=ny * nx, nt=nt
    )
    model_C = HeatFluxNet(
        strategy="modal",
        hidden_layer_sizes=(192, 192, 192),
        activation="tanh",
        alpha=6e-4,
        learning_rate_init=7e-4,
        max_iter=100,
    )
    model_C.fit(X_C_tr, y_C_tr, n_samples=50_000)

    # Reconstruct: sum predicted contributions across modes
    y_C_pred = model_C.predict(X_C_te)
    nt_test = nt - int(0.7 * nt)
    q_c_pred_flat = y_C_pred.sum(axis=1)
    q_c_pred = q_c_pred_flat.reshape(ny * nx, nt_test)
    q_pred_field = (q_c_pred + q_mean_vec[:, None]).reshape(-1)
    nt_train = int(0.7 * nt)
    q_true_test = q.reshape(ny * nx, nt)[:, nt_train:].reshape(-1)
    m_C = evaluate(q_true_test, q_pred_field, split="test")
    print(f"    {m_C}")

    # ── 8. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results comparison")
    print("=" * 60)
    print(f"  DMD      R² = {r2_dmd:.4f}")
    print(f"  Model A  {m_A}")
    print(f"  Model B  {m_B}")
    print(f"  Model C  {m_C}")

    # ── 9. Save a summary plot ────────────────────────────────────────────────
    print("\n[8] Saving plots...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("icarus quickstart — POD analysis")

    plot_cumulative_energy(pod_T, ax=axes[0])
    axes[0].set_title("Temperature cumulative POD energy")

    mode0 = pod_T.modes_[:, 0].reshape(ny, nx)
    axes[1].imshow(mode0, cmap="RdBu_r", origin="upper",
                   vmin=-np.abs(mode0).max(), vmax=np.abs(mode0).max())
    axes[1].set_title("Temperature mode 1 (spatial)")
    axes[1].set_xlabel("x pixels")
    axes[1].set_ylabel("y pixels")

    axes[2].plot(pod_T.temporal_coefficients_[0], lw=0.8, color="steelblue",
                 label="Temperature")
    axes[2].plot(pod_q.temporal_coefficients_[0], lw=0.8, color="darkorange",
                 alpha=0.8, label="Heat flux")
    axes[2].set_title("Mode 1 temporal coefficients (normalised)")
    axes[2].set_xlabel("Time steps")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "icarus_quickstart.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"    Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
