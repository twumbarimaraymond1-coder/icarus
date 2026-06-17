"""
examples/spod_analysis.py
=========================
Spectral POD diagnostic of a boiling dataset — Quickstart style.

All the heavy lifting (low-memory loading, cropping/centring, the SPOD itself,
and plotting) lives in the icarus package, so this script is just a few calls.

    pip install -U icarus-thermal
    python examples/spod_analysis.py --file "path/to/MODEL_~1.MAT" --field heatflux
"""

import argparse

import icarus as tf


def main(args):
    # 1. Load one field, low-memory (only the heater layer is read off disk).
    field, dt = tf.load_field(args.file, args.field)

    # 2. Fit Spectral POD straight from the [ny, nx, nt] field.
    spod = tf.SPOD(block_size=args.block_size).fit_field(
        field, dt=dt, spatial_crop=args.spatial_crop, trim_frames=args.trim_frames)

    # 3. Report and plot.
    print("Dominant frequencies (Hz):", spod.dominant_frequencies(n=args.n_peaks))
    spod.plot_spectrum(path=f"spod_spectrum_{args.field}.png")
    for freq in spod.dominant_frequencies(n=args.n_peaks):
        spod.plot_mode(freq, path=f"spod_mode_{freq:.0f}Hz_{args.field}.png")

    print(f"Done — wrote spod_spectrum_{args.field}.png and per-peak mode maps.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Spectral POD analysis of a boiling dataset.")
    p.add_argument("--file", required=True, help="Path to a .mat / .npz dataset")
    p.add_argument("--field", choices=["temp", "heatflux"], default="heatflux")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--n-peaks", type=int, default=4)
    p.add_argument("--spatial-crop", type=int, default=5)
    p.add_argument("--trim-frames", type=int, default=43)
    main(p.parse_args())
