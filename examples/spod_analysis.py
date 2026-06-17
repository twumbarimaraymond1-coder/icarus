"""
examples/spod_analysis.py
=========================
Diagnostic Spectral POD (SPOD) analysis of a single boiling dataset.

The question this answers: **do boiling mechanisms show up as distinct
frequency peaks?** If single-phase convection, bubble nucleation/departure and
microlayer dynamics live at separable timescales, the SPOD energy spectrum will
show distinct bumps, and the leading SPOD mode at each peak reveals that
mechanism's spatial fingerprint. That single plot is the evidence for the
SPOD -> heat-partition through-line.

Memory-friendly by design
-------------------------
The temperature variable in the real .mat files is 4-D [ny, nx, nz, nt] (~2 GB
if fully loaded). This script reads **only the heater z-layer (index 0)**
directly off disk with h5py, so it needs ~200 MB, not ~2 GB. The heat-flux
field (qL2) is already small. Both can run on a modest laptop.

Usage
-----
    # Heat-flux field (lightest; the mechanism signatures live here)
    python examples/spod_analysis.py --file path/to/MODEL_~1.MAT --field heatflux

    # Temperature field (heater surface only)
    python examples/spod_analysis.py --file path/to/MODEL_~1.MAT --field temp

    # Tune the time/frequency trade-off and trim startup frames
    python examples/spod_analysis.py --file data.mat --block-size 512 \
        --overlap 0.5 --trim-frames 43 --spatial-crop 5 --out spod_out

Outputs (into --out): a spectrum PNG, leading-mode PNGs at the top peaks, and a
spod_summary.json with the peak frequencies and energies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless: write PNGs, no display needed
import matplotlib.pyplot as plt

from icarus.decomposition.spod import SPOD


# ── Low-memory field loader ───────────────────────────────────────────────────

def load_field_lowmem(path: Path, field: str,
                      temp_key: str = "T", heatflux_key: str = "qL2",
                      timestep_key: str = "TimeStep") -> tuple[np.ndarray, float]:
    """Load one field as [ny, nx, nt] + dt, reading as little as possible.

    For HDF5/v7.3 .mat: MATLAB stores arrays axis-reversed. A 4-D temperature
    array [ny, nx, nz, nt] is stored [nt, nz, nx, ny]; we slice z-index 0 off
    disk (axis 1) so only one layer is read. .npz falls back to numpy.
    """
    key = temp_key if field == "temp" else heatflux_key
    suffix = path.suffix.lower()

    if suffix in (".mat", ".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"'{key}' not in {path}. Keys: {list(f.keys())}")
            ds = f[key]
            if ds.ndim == 4:
                # stored [nt, nz, nx, ny] -> take heater layer 0 -> [nt, nx, ny]
                arr = np.asarray(ds[:, 0, :, :], dtype=np.float64)
            elif ds.ndim == 3:
                arr = np.asarray(ds[...], dtype=np.float64)  # [nt, nx, ny]
            else:
                raise ValueError(f"Unexpected ndim={ds.ndim} for '{key}'.")
            field_arr = arr.transpose(2, 1, 0)               # -> [ny, nx, nt]
            dt = float(np.array(f[timestep_key]).flat[0]) if timestep_key in f else 1.0
        return field_arr, dt

    # .npz / .npy
    archive = np.load(path)
    field_arr = np.asarray(archive[key], dtype=np.float64)
    if field_arr.ndim == 4:
        field_arr = field_arr[:, :, 0, :]
    dt = float(archive[timestep_key]) if timestep_key in archive else 1.0
    return field_arr, dt


# ── Preprocess (crop, trim, mean-centre) ──────────────────────────────────────

def preprocess(field: np.ndarray, spatial_crop: int, trim_frames: int
               ) -> np.ndarray:
    """Crop borders, trim startup frames, mean-centre per pixel -> [n_pix, nt]."""
    c = spatial_crop
    if c > 0:
        field = field[c:-c, c:-c, :]
    if trim_frames > 0:
        field = field[:, :, trim_frames:]
    ny, nx, nt = field.shape
    X = field.reshape(ny * nx, nt)
    X_c = X - X.mean(axis=1, keepdims=True)     # remove per-pixel time mean
    return X_c, ny, nx


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.field} field from {path.name} ...")
    field, dt = load_field_lowmem(path, args.field)
    print(f"  raw shape {field.shape}, dt = {dt:g} s  (fs = {1/dt:,.1f} Hz)")

    X_c, ny, nx = preprocess(field, args.spatial_crop, args.trim_frames)
    n_pix, nt = X_c.shape
    print(f"  after crop/trim: {ny}x{nx} pixels, {nt} frames")
    del field

    if args.block_size > nt:
        raise ValueError(f"--block-size {args.block_size} > available frames {nt}. "
                         f"Use a smaller block.")

    print(f"Running SPOD (block_size={args.block_size}, overlap={args.overlap}) ...")
    spod = SPOD(n_modes=args.n_modes, block_size=args.block_size,
                overlap=args.overlap).fit(X_c, dt=dt)
    f = spod.frequencies_
    lead = spod.spectrum(0)                      # leading-mode energy vs freq
    print(f"  {spod.n_blocks_} Welch blocks, df = {f[1]-f[0]:.3g} Hz, "
          f"f_max = {f[-1]:.3g} Hz")

    # Peak frequencies come straight from the package; map to bin indices.
    peak_freqs = spod.dominant_frequencies(n=args.n_peaks)
    peaks = [int(np.argmin(np.abs(f - pf))) for pf in peak_freqs]
    print("\nTop SPOD energy peaks (leading mode):")
    for rank, k in enumerate(peaks, 1):
        print(f"  {rank}. f = {f[k]:8.2f} Hz   energy = {lead[k]:.4e}")

    # ── Spectrum plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in range(min(3, spod.n_modes_retained_)):
        ax.semilogy(f, spod.spectrum(m), lw=1.3, label=f"mode {m+1}")
    for k in peaks:
        ax.axvline(f[k], color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.annotate(f"{f[k]:.1f} Hz", (f[k], lead[k]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPOD modal energy")
    ax.set_title(f"SPOD spectrum — {args.field} — {path.name}")
    ax.legend()
    fig.tight_layout()
    spec_png = out / f"spod_spectrum_{args.field}.png"
    fig.savefig(spec_png, dpi=130)
    plt.close(fig)
    print(f"\nSpectrum -> {spec_png}")

    # ── Leading mode at each peak ─────────────────────────────────────────────
    for rank, k in enumerate(peaks, 1):
        mode = spod.modes_[k, :, 0].reshape(ny, nx)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(np.abs(mode), cmap="inferno", origin="lower")
        ax.set_title(f"Leading SPOD mode @ {f[k]:.1f} Hz  ({args.field})")
        fig.colorbar(im, ax=ax, label="|mode| (coherent amplitude)")
        fig.tight_layout()
        mode_png = out / f"spod_mode_peak{rank}_{f[k]:.0f}Hz_{args.field}.png"
        fig.savefig(mode_png, dpi=130)
        plt.close(fig)
        print(f"Mode @ {f[k]:.1f} Hz -> {mode_png}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "file": path.name, "field": args.field, "dt": dt,
        "ny": ny, "nx": nx, "nt": nt,
        "block_size": args.block_size, "overlap": args.overlap,
        "n_blocks": spod.n_blocks_, "df_hz": float(f[1] - f[0]),
        "f_max_hz": float(f[-1]),
        "peaks": [{"rank": r, "freq_hz": float(f[k]),
                   "leading_energy": float(lead[k])}
                  for r, k in enumerate(peaks, 1)],
    }
    (out / f"spod_summary_{args.field}.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {out / ('spod_summary_' + args.field + '.json')}")
    print("\nInterpretation: distinct peaks at separated frequencies = distinct "
          "timescales = candidate physical mechanisms. Compare the mode maps "
          "across peaks to see their spatial signatures.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="Path to a .mat / .h5 / .npz dataset")
    p.add_argument("--field", choices=["temp", "heatflux"], default="heatflux",
                   help="Which field to analyse (default: heatflux — lightest)")
    p.add_argument("--block-size", type=int, default=512,
                   help="Welch block length in frames (freq resolution vs convergence)")
    p.add_argument("--overlap", type=float, default=0.5, help="Block overlap fraction")
    p.add_argument("--n-modes", type=int, default=3, help="SPOD modes per frequency")
    p.add_argument("--n-peaks", type=int, default=4, help="How many peaks to plot")
    p.add_argument("--spatial-crop", type=int, default=5, help="Border pixels to drop")
    p.add_argument("--trim-frames", type=int, default=43, help="Startup frames to drop")
    p.add_argument("--out", default="spod_out", help="Output folder")
    main(p.parse_args())
