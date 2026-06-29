"""
examples/bandwise_stochastic.py
===============================
Timescale-resolved, uncertainty-aware heat-flux prediction — Quickstart style.

Partitions temperature and heat flux into frequency bands, fits a Model-C modal
mapping per band as a deep ensemble, and predicts the heat-flux field with a
calibrated uncertainty band.

    pip install -U icarus-thermal
    python examples/bandwise_stochastic.py --file "path/to/MODEL_~1.MAT"
"""

import argparse

import icarus as tf


def main(args):
    T, _ = tf.load_field(args.file, "temp")
    q, dt = tf.load_field(args.file, "heatflux")

    model = tf.BandwiseModalModel(
        edges=args.edges, n_pod_modes=args.n_pod_modes, n_members=args.n_members,
        n_training_samples=args.n_samples,
        model_kwargs={"hidden_layer_sizes": (64,), "max_iter": 150})
    model.fit(T, q, dt=dt, spatial_crop=args.spatial_crop, trim_frames=args.trim_frames)

    print("Per-band and total R²:")
    model.evaluate()

    Tc = T[args.spatial_crop:-args.spatial_crop,
           args.spatial_crop:-args.spatial_crop, args.trim_frames:]
    qc = q[args.spatial_crop:-args.spatial_crop,
           args.spatial_crop:-args.spatial_crop, args.trim_frames:]
    _, lower, upper = model.predict_interval(Tc, coverage=0.9)
    cal = tf.interval_metrics(qc, lower, upper)
    print(f"\n90% interval -> coverage {cal['coverage']:.3f}, "
          f"mean width {cal['mean_width']:,.0f} W/m^2")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--edges", type=float, nargs="+", default=[200, 1000])
    p.add_argument("--n-pod-modes", type=int, default=5)
    p.add_argument("--n-members", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=120000)
    p.add_argument("--spatial-crop", type=int, default=5)
    p.add_argument("--trim-frames", type=int, default=43)
    main(p.parse_args())
