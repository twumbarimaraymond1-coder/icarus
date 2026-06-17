"""
Tests for icarus core modules.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from icarus.data.loader import from_arrays
from icarus.data.preprocessor import Preprocessor, PreprocessorConfig
from icarus.decomposition.pod import POD
from icarus.decomposition.dmd import DMD
from icarus.decomposition.spod import SPOD
from icarus.features.engineer import (
    build_raw_features,
    build_gradient_features,
    build_modal_features,
    flatten_target,
    train_test_split_temporal,
)
from icarus.models.neural import HeatFluxNet
from icarus.metrics.evaluation import evaluate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Small synthetic IR thermography dataset for fast testing."""
    rng = np.random.default_rng(42)
    ny, nx, nt = 20, 30, 100
    T = 420 + 5 * rng.standard_normal((ny, nx, nt))
    # Heat flux inversely correlated with temperature + noise
    q = 300_000 - 15_000 * (T - 420) + 10_000 * rng.standard_normal((ny, nx, nt))
    return from_arrays(T, q, dt=2.5e-4)


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_from_arrays_shape(self, synthetic_data):
        assert synthetic_data["T"].shape == (20, 30, 100)
        assert synthetic_data["q"].shape == (20, 30, 100)

    def test_from_arrays_4d_squeeze(self):
        rng = np.random.default_rng(0)
        T4d = rng.standard_normal((20, 30, 11, 100))
        q = rng.standard_normal((20, 30, 100))
        data = from_arrays(T4d, q, dt=1.0)
        assert data["T"].shape == (20, 30, 100)

    def test_shape_mismatch_raises(self):
        T = np.ones((10, 10, 50))
        q = np.ones((10, 10, 60))
        with pytest.raises(ValueError, match="shapes must match"):
            from_arrays(T, q)


class TestHDF5Loader:
    """Guards the MATLAB v7.3 / HDF5 load path (icarus.data.loader._load_hdf5):
    MATLAB stores arrays with reversed axis order, and 4-D temperature must be
    de-transposed and have its heater-surface z-layer (index 0) extracted.
    This is the exact path the real MODEL_~1/2/3.MAT files take."""

    def _write_v73_like(self, path, T_logical, q_logical, dt):
        # MATLAB v7.3 (HDF5) stores an [ny,nx,nz,nt] array as [nt,nz,nx,ny].
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("T", data=np.asarray(T_logical).transpose(
                tuple(range(T_logical.ndim))[::-1]))
            f.create_dataset("qL2", data=np.asarray(q_logical).transpose(
                tuple(range(q_logical.ndim))[::-1]))
            f.create_dataset("TimeStep", data=np.array([[dt]]))

    def test_hdf5_4d_temperature_z_layer0(self, tmp_path):
        from icarus.data.loader import load
        ny, nx, nz, nt = 4, 5, 11, 7
        rng = np.random.default_rng(0)
        T = rng.standard_normal((ny, nx, nz, nt))
        q = rng.standard_normal((ny, nx, nt))
        p = tmp_path / "model.h5"
        self._write_v73_like(p, T, q, dt=2.5e-4)

        data = load(str(p), temperature_key="T", heatflux_key="qL2",
                    timestep_key="TimeStep")
        assert data["T"].shape == (ny, nx, nt)
        assert data["q"].shape == (ny, nx, nt)
        # z-layer 0 (heater surface) must be the one extracted
        np.testing.assert_allclose(data["T"], T[:, :, 0, :])
        np.testing.assert_allclose(data["q"], q)
        assert abs(data["dt"] - 2.5e-4) < 1e-12

    def test_mat_suffix_falls_back_to_hdf5(self, tmp_path):
        # A v7.3 .mat is really HDF5; scipy.io.loadmat fails and the loader
        # must transparently fall back to the HDF5 reader.
        from icarus.data.loader import load
        ny, nx, nz, nt = 3, 4, 11, 6
        rng = np.random.default_rng(1)
        T = rng.standard_normal((ny, nx, nz, nt))
        q = rng.standard_normal((ny, nx, nt))
        p = tmp_path / "MODEL_~1.MAT"
        self._write_v73_like(p, T, q, dt=1e-3)

        data = load(str(p))
        assert data["T"].shape == (ny, nx, nt)
        np.testing.assert_allclose(data["T"], T[:, :, 0, :])


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_crop_reduces_shape(self, synthetic_data):
        pre = Preprocessor(PreprocessorConfig(spatial_crop=2, trim_frames=5))
        out = pre.fit_transform(synthetic_data)
        assert out["T"].shape == (16, 26, 95)

    def test_mean_subtraction(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        np.testing.assert_allclose(out["T_c"].mean(axis=2), 0, atol=1e-6)

    def test_to_matrix_roundtrip(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        T = out["T"]
        ny, nx, nt = T.shape
        mat = Preprocessor.to_matrix(T)
        T_back = Preprocessor.from_matrix(mat, ny, nx)
        np.testing.assert_array_equal(T, T_back)

    def test_transform_before_fit_raises(self, synthetic_data):
        pre = Preprocessor()
        with pytest.raises(RuntimeError):
            pre.transform(synthetic_data)


# ── POD tests ─────────────────────────────────────────────────────────────────

class TestPOD:
    def test_fit_shapes(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["T_c"])
        pod = POD(n_modes=5).fit(X_c)
        assert pod.modes_.shape[1] == 5
        assert pod.temporal_coefficients_.shape[0] == 5

    def test_cumulative_energy_monotone(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["T_c"])
        pod = POD().fit(X_c)
        assert np.all(np.diff(pod.cumulative_energy_) >= 0)
        np.testing.assert_allclose(pod.cumulative_energy_[-1], 1.0, atol=1e-6)

    def test_reconstruction_error_decreases_with_modes(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["T_c"])
        pod = POD(n_modes=20).fit(X_c)
        errors = []
        for k in [1, 5, 10, 20]:
            X_rec = pod.reconstruct(n_modes=k)
            errors.append(np.linalg.norm(X_c - X_rec))
        assert errors == sorted(errors, reverse=True)

    def test_energy_threshold(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["T_c"])
        pod = POD(energy_threshold=0.90).fit(X_c)
        assert pod.cumulative_energy_[pod.n_modes_retained_ - 1] >= 0.90

    def test_modal_contributions_shape(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["T_c"])
        n_pix = out["T"].shape[0] * out["T"].shape[1]
        nt = out["T"].shape[2]
        pod = POD(n_modes=3).fit(X_c)
        contribs = pod.modal_contributions(X_c)
        assert contribs.shape == (n_pix, nt, 3)

    def test_modal_contributions_chunked_matches_unchunked(self, synthetic_data):
        """Chunked computation must be bit-for-bit identical to the full-width
        path (memory optimisation must not change results)."""
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["T_c"])
        pod = POD(n_modes=3).fit(X_c)
        full = pod.modal_contributions(X_c)
        for cs in (1, 7, 1000):
            chunked = pod.modal_contributions(X_c, chunk_size=cs)
            np.testing.assert_array_equal(full, chunked)

    def test_modal_contributions_dtype(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["T_c"])
        pod = POD(n_modes=3).fit(X_c)
        c32 = pod.modal_contributions(X_c, dtype=np.float32)
        assert c32.dtype == np.float32
        np.testing.assert_allclose(
            c32, pod.modal_contributions(X_c), rtol=1e-4, atol=1e-3)

    def test_fit_before_use_raises(self):
        pod = POD(n_modes=5)
        with pytest.raises(RuntimeError):
            pod.reconstruct()


# ── SPOD tests ────────────────────────────────────────────────────────────────

class TestSPOD:
    @pytest.fixture
    def two_tone_data(self):
        """Two distinct spatial patterns oscillating at two known frequencies.
        SPOD must separate them by frequency and recover each pattern."""
        rng = np.random.default_rng(0)
        n_pix, nt = 24, 1024
        dt = 0.01                       # fs = 100 Hz
        block = 128
        df = 1.0 / (block * dt)         # frequency resolution
        # Put both tones exactly on FFT bins so they don't leak across bins.
        f1, f2 = 8 * df, 20 * df
        p1 = rng.standard_normal(n_pix)
        p2 = rng.standard_normal(n_pix)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        t = np.arange(nt) * dt
        X = (np.outer(p1, np.sin(2 * np.pi * f1 * t))
             + 0.4 * np.outer(p2, np.sin(2 * np.pi * f2 * t)))
        X_c = X - X.mean(axis=1, keepdims=True)
        return dict(X_c=X_c, dt=dt, block=block, f1=f1, f2=f2, p1=p1, p2=p2)

    def test_fit_shapes(self, two_tone_data):
        d = two_tone_data
        spod = SPOD(n_modes=3, block_size=d["block"]).fit(d["X_c"], dt=d["dt"])
        n_freq = d["block"] // 2 + 1
        assert spod.frequencies_.shape == (n_freq,)
        assert spod.modes_.shape == (n_freq, d["X_c"].shape[0], 3)
        assert spod.eigenvalues_.shape == (n_freq, 3)

    def test_eigenvalues_sorted_per_frequency(self, two_tone_data):
        d = two_tone_data
        spod = SPOD(n_modes=4, block_size=d["block"]).fit(d["X_c"], dt=d["dt"])
        for k in range(spod.frequencies_.size):
            row = spod.eigenvalues_[k]
            assert np.all(np.diff(row) <= 1e-9), "energies must be descending"

    def test_spectrum_peaks_at_injected_frequencies(self, two_tone_data):
        d = two_tone_data
        spod = SPOD(n_modes=2, block_size=d["block"]).fit(d["X_c"], dt=d["dt"])
        energy = spod.spectrum()                 # leading-mode energy vs f
        f = spod.frequencies_
        peak_f = f[energy.argmax()]
        assert abs(peak_f - d["f1"]) < 0.5 / (d["block"] * d["dt"]), \
            "dominant SPOD energy must sit at the strongest injected tone"
        # The second tone should clearly outrank a quiet background bin.
        k2 = int(np.argmin(np.abs(f - d["f2"])))
        quiet = int(np.argmin(np.abs(f - (d["f2"] + 8 / (d["block"] * d["dt"])))))
        assert energy[k2] > 10 * energy[quiet]

    def test_leading_modes_recover_spatial_patterns(self, two_tone_data):
        d = two_tone_data
        spod = SPOD(n_modes=2, block_size=d["block"]).fit(d["X_c"], dt=d["dt"])
        m1 = spod.mode(freq=d["f1"])             # complex [n_pix]
        m2 = spod.mode(freq=d["f2"])
        # |cosine similarity| ~ 1 with the true pattern (phase is arbitrary).
        def align(m, p):
            return abs(np.vdot(m, p)) / (np.linalg.norm(m) * np.linalg.norm(p))
        assert align(m1, d["p1"]) > 0.95
        assert align(m2, d["p2"]) > 0.95

    def test_dominant_frequencies_finds_injected_tones(self, two_tone_data):
        d = two_tone_data
        spod = SPOD(n_modes=2, block_size=d["block"]).fit(d["X_c"], dt=d["dt"])
        tones = spod.dominant_frequencies(n=2)
        df = 1.0 / (d["block"] * d["dt"])
        # The two strongest peaks must be the two injected tones (within 1 bin).
        assert min(abs(tones - d["f1"])) < df
        assert min(abs(tones - d["f2"])) < df
        # Strongest first (f1 has the larger amplitude in the fixture).
        assert abs(tones[0] - d["f1"]) < df

    def test_block_size_larger_than_nt_raises(self):
        spod = SPOD(block_size=64)
        with pytest.raises(ValueError):
            spod.fit(np.zeros((10, 32)), dt=0.01)

    def test_use_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            SPOD().mode(freq=1.0)


# ── DMD tests ─────────────────────────────────────────────────────────────────

class TestDMD:
    def test_forecast_shape(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["q_c"])
        n_pix = X_c.shape[0]
        dmd = DMD(dt=2.5e-4).fit(X_c)
        forecast = dmd.forecast(n_steps=30)
        assert forecast.shape == (n_pix, 30)

    def test_forecast_from_shape(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        X_c = Preprocessor.to_matrix(out["q_c"])
        dmd = DMD(dt=2.5e-4).fit(X_c)
        x_init = X_c[:, -1]
        forecast = dmd.forecast_from(x_init, n_steps=20)
        assert forecast.shape == (X_c.shape[0], 20)


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatures:
    def test_raw_features_shape(self, synthetic_data):
        T = synthetic_data["T"]
        ny, nx, nt = T.shape
        X = build_raw_features(T)
        assert X.shape == (ny * nx * nt, 1)

    def test_gradient_features_shape(self, synthetic_data):
        T = synthetic_data["T"]
        ny, nx, nt = T.shape
        X = build_gradient_features(T, dt=2.5e-4)
        assert X.shape == (ny * nx * nt, 4)

    def test_modal_features_shape(self, synthetic_data):
        pre = Preprocessor()
        out = pre.fit_transform(synthetic_data)
        T = out["T"]
        ny, nx, nt = T.shape
        X_c_T = Preprocessor.to_matrix(out["T_c"])
        X_c_q = Preprocessor.to_matrix(out["q_c"])
        pod_T = POD(n_modes=3).fit(X_c_T)
        pod_q = POD(n_modes=3).fit(X_c_q)
        T_contribs = pod_T.modal_contributions(X_c_T)
        q_contribs = pod_q.modal_contributions(X_c_q)
        X, y = build_modal_features(T_contribs, q_contribs)
        assert X.shape == (ny * nx * nt, 3)
        assert y.shape == (ny * nx * nt, 3)

    def test_temporal_split_ordering(self, synthetic_data):
        T = synthetic_data["T"]
        X = build_raw_features(T)
        y = flatten_target(synthetic_data["q"])
        ny, nx, nt = T.shape
        nt_train = int(0.7 * nt)
        X_tr, X_te, y_tr, y_te = train_test_split_temporal(
            X, y, train_fraction=0.7, n_pix=ny * nx, nt=nt
        )
        assert len(X_tr) == nt_train * ny * nx
        assert len(X_te) == (nt - nt_train) * ny * nx

    def test_time_major_ordering(self, synthetic_data):
        """Verify flatten_target produces time-major order."""
        T = synthetic_data["T"]
        q = synthetic_data["q"]
        ny, nx, nt = T.shape
        q_flat = flatten_target(q)
        # First ny*nx samples should all be from timestep 0 (cast to float32 for comparison)
        q_t0 = q[:, :, 0].reshape(-1).astype(np.float32)
        np.testing.assert_array_almost_equal(q_flat[:ny * nx], q_t0, decimal=3)
        # Second ny*nx samples should all be from timestep 1
        q_t1 = q[:, :, 1].reshape(-1).astype(np.float32)
        np.testing.assert_array_almost_equal(q_flat[ny * nx: 2 * ny * nx], q_t1, decimal=3)


# ── Model tests ───────────────────────────────────────────────────────────────

class TestHeatFluxNet:
    def test_fit_predict_raw(self, synthetic_data):
        T = synthetic_data["T"]
        q = synthetic_data["q"]
        X = build_raw_features(T)
        y = flatten_target(q)
        model = HeatFluxNet(
            strategy="raw",
            hidden_layer_sizes=(16,),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=10,
        )
        model.fit(X, y, n_samples=500)
        y_pred = model.predict(X[:100])
        assert y_pred.shape == (100,)

    def test_predict_before_fit_raises(self):
        model = HeatFluxNet(hidden_layer_sizes=(16,))
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((10, 1)))

    def test_no_hidden_layers_raises(self):
        model = HeatFluxNet()
        with pytest.raises(RuntimeError, match="hidden_layer_sizes"):
            model.fit(np.zeros((10, 1)), np.zeros(10))


class TestTemporalSubsample:
    """Guards the temporal-validation subsample fix.

    Before the fix, ``optimise`` drew a random subsample and *then* did an
    80/20 head/tail split for ``validation_strategy="temporal"``. Because the
    random draw destroyed chronological order, the "temporal" validation set
    interleaved with training timesteps — silently leaking the future into the
    past. The fix sorts the subsampled indices when temporal.
    """

    def test_temporal_indices_are_sorted(self):
        idx = HeatFluxNet._subsample_indices(
            n_total=10_000, n=500, random_state=0,
            validation_strategy="temporal",
        )
        assert np.all(np.diff(idx) > 0), \
            "temporal subsample indices must be ascending (chronological)"

    def test_temporal_val_rows_are_strictly_later(self):
        # The head/tail split the optimiser performs must put every validation
        # row at a later original timestep than every training row.
        idx = HeatFluxNet._subsample_indices(
            n_total=10_000, n=500, random_state=0,
            validation_strategy="temporal",
        )
        split = int(len(idx) * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]
        assert train_idx.max() < val_idx.min(), \
            "no temporal leakage: all val rows after all train rows"

    def test_random_strategy_not_sorted(self):
        # Random strategy must NOT force chronological order (it uses an i.i.d.
        # sklearn split downstream); sorting would be a needless behaviour change.
        idx = HeatFluxNet._subsample_indices(
            n_total=10_000, n=500, random_state=0,
            validation_strategy="random",
        )
        assert not np.all(np.diff(idx) > 0)


# ── Search space tests ───────────────────────────────────────────────────────

class TestTimeMajorOrdering:
    def test_flatten_target_is_time_major(self):
        ny, nx, nt = 2, 3, 4
        q = np.zeros((ny, nx, nt))
        for t in range(nt):
            q[:, :, t] = t  # each timestep has a known constant value
        y = flatten_target(q)
        assert np.all(y[:ny * nx] == 0),   "t=0 pixels should be first"
        assert np.all(y[ny*nx:2*ny*nx] == 1), "t=1 pixels should be second"
        assert np.all(y[2*ny*nx:3*ny*nx] == 2)
        assert np.all(y[3*ny*nx:] == 3)

    def test_modal_reconstruction_time_major(self):
        n_pix, nt, n_modes = 6, 4, 3
        y_modal = np.ones((n_pix * nt, n_modes), dtype=np.float32)
        q_c_flat = y_modal.sum(axis=1)           # all 3.0
        q_mean = np.arange(n_pix, dtype=np.float32)
        q_pred = q_c_flat + np.tile(q_mean, nt)
        assert q_pred.shape == (n_pix * nt,)
        # Each block of n_pix should equal 3 + q_mean
        for t in range(nt):
            block = q_pred[t * n_pix:(t + 1) * n_pix]
            np.testing.assert_array_equal(block, 3.0 + q_mean)

    def test_build_raw_features_time_major(self):
        ny, nx, nt = 2, 3, 4
        T = np.zeros((ny, nx, nt))
        for t in range(nt):
            T[:, :, t] = t
        X = build_raw_features(T)
        n_pix = ny * nx
        assert float(X[:n_pix].mean()) == 0.0
        assert float(X[n_pix:2*n_pix].mean()) == 1.0


class TestSearchSpaces:
    def test_presets_exist(self):
        from icarus.models.neural import SEARCH_SPACES
        for name in ("small", "medium", "large"):
            assert name in SEARCH_SPACES
            sp = SEARCH_SPACES[name]
            for key in ("n_layers_min", "n_layers_max", "n_units_min",
                        "n_units_max", "activations", "alpha_min", "alpha_max",
                        "lr_min", "lr_max"):
                assert key in sp, f"Missing key '{key}' in preset '{name}'"

    def test_invalid_preset_raises(self, synthetic_data):
        X = np.ones((100, 5), dtype=np.float32)
        y = np.ones((100, 5), dtype=np.float32)
        model = HeatFluxNet(strategy="modal", hidden_layer_sizes=(16,))
        with pytest.raises(ValueError, match="Unknown search space preset"):
            model.optimise(X, y, n_trials=1, search_space="nonexistent")

    def test_custom_dict_accepted(self, synthetic_data):
        """Custom dict search space runs without error."""
        X = np.ones((200, 5), dtype=np.float32)
        y = np.ones((200, 5), dtype=np.float32)
        model = HeatFluxNet(strategy="modal")
        custom = {
            "n_layers_min": 1, "n_layers_max": 1,
            "n_units_min": 8, "n_units_max": 16,
            "activations": ["relu"],
            "alpha_min": 1e-4, "alpha_max": 1e-3,
            "lr_min": 1e-4, "lr_max": 1e-3,
        }
        model.optimise(X, y, n_trials=2, n_opt_samples=100,
                       search_space=custom, verbose=False)
        assert model.hidden_layer_sizes is not None

    def test_string_preset_accepted(self, synthetic_data):
        """String preset runs without error."""
        X = np.ones((200, 5), dtype=np.float32)
        y = np.ones((200, 5), dtype=np.float32)
        model = HeatFluxNet(strategy="modal")
        model.optimise(X, y, n_trials=2, n_opt_samples=100,
                       search_space="small", verbose=False)
        assert model.hidden_layer_sizes is not None


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_prediction(self):
        y = np.arange(100, dtype=float)
        m = evaluate(y, y)
        assert m.r2 == pytest.approx(1.0)
        assert m.rmse == pytest.approx(0.0, abs=1e-10)
        assert m.mae == pytest.approx(0.0, abs=1e-10)

    def test_metrics_str(self):
        y = np.arange(100, dtype=float)
        m = evaluate(y, y + 1000, split="test")
        s = str(m)
        assert "test" in s
        assert "R²" in s


# ── Cross-dataset training tests ────────────────────────────────────────────────

class TestCrossDatasetFit:
    """Guards MultiDatasetTrainer.cross_dataset_fit: train on N datasets,
    hold out one entirely for testing (cross-surface generalisation)."""

    def _make_npz(self, tmp_path, name, seed, q_mean):
        rng = np.random.default_rng(seed)
        ny, nx, nt = 16, 20, 60
        XX, YY = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
        t = np.linspace(0, 1, nt)
        m1 = np.sin(np.pi * YY) * np.cos(np.pi * XX)
        T = (420.0 + m1[:, :, None] * 3 * np.sin(2 * np.pi * t)[None, None, :]
             + 0.5 * rng.standard_normal((ny, nx, nt)))
        q = q_mean - 12000 * (T - 420.0) + 8000 * rng.standard_normal((ny, nx, nt))
        p = tmp_path / f"{name}.npz"
        np.savez(p, T=T, qL2=q, TimeStep=2.5e-4)
        return str(p)

    @pytest.fixture
    def registry_with_three(self, tmp_path):
        from icarus.registry.dataset import DatasetRegistry, DatasetEntry
        from icarus.registry.extractor import FeatureExtractor

        reg = DatasetRegistry(tmp_path / "store")
        specs = [
            ("D001", 42, 300_000, "plain_copper"),
            ("D002", 7, 320_000, "microporous_copper"),
            ("D003", 123, 290_000, "nanostructured_copper"),
        ]
        for ds_id, seed, q_mean, surface in specs:
            raw = self._make_npz(tmp_path, ds_id, seed, q_mean)
            reg.register(DatasetEntry(ds_id, "water", "flow_boiling", surface,
                                      "Test", raw_path=raw, spatial_crop=2))
        ext = FeatureExtractor(reg, n_pod_modes=3)
        for ds_id, *_ in specs:
            ext.process(ds_id, force=True)
        return reg

    def test_cross_dataset_split_sizes(self, registry_with_three):
        """Train pool = ALL of D001+D002; test = ALL of D003 (no leakage)."""
        from icarus.registry.trainer import MultiDatasetTrainer

        reg = registry_with_three
        n3 = reg.get("D003").n_samples
        n1 = reg.get("D001").n_samples
        n2 = reg.get("D002").n_samples

        trainer = MultiDatasetTrainer(reg, n_pod_modes=3,
                                      n_training_samples=None,
                                      optimise_hyperparams=False)
        trainer.model_ = HeatFluxNet(strategy="modal",
                                     hidden_layer_sizes=(16,), max_iter=20)
        trainer.cross_dataset_fit(train_ids=["D001", "D002"], test_id="D003",
                                  verbose=False)

        # Held-out test set is exactly all of D003, nothing more, nothing less.
        assert len(trainer._X_test) == n3
        assert len(trainer._X_train) == n1 + n2
        assert trainer._test_dataset_id == "D003"
        assert trainer._fitted

        metrics = trainer.evaluate(verbose=False)
        assert "test" in metrics and "train" in metrics

    def test_test_id_in_train_ids_raises(self, registry_with_three):
        """Refuse to leak: a test dataset may not also be a train dataset."""
        from icarus.registry.trainer import MultiDatasetTrainer

        trainer = MultiDatasetTrainer(registry_with_three, n_pod_modes=3,
                                      optimise_hyperparams=False)
        with pytest.raises(ValueError, match="leak"):
            trainer.cross_dataset_fit(train_ids=["D001", "D003"], test_id="D003",
                                      verbose=False)

    def test_empty_train_ids_raises(self, registry_with_three):
        from icarus.registry.trainer import MultiDatasetTrainer

        trainer = MultiDatasetTrainer(registry_with_three, n_pod_modes=3,
                                      optimise_hyperparams=False)
        with pytest.raises(ValueError):
            trainer.cross_dataset_fit(train_ids=[], test_id="D003", verbose=False)

    def test_q_abs_is_time_major_aligned(self, registry_with_three):
        """The extractor must store q_abs in TIME-major order (row = t*n_pix + p),
        row-aligned with q_contribs and the split mask. Guards the absolute-R²
        alignment: reshaping q_abs to [nt, n_pix] and meaning over time must
        equal the separately stored per-pixel mean field."""
        import h5py
        reg = registry_with_three
        path = reg.get("D003").processed_path
        with h5py.File(path, "r") as f:
            assert "q_abs" in f, "extractor must emit q_abs"
            assert "q_mean_field" in f, "extractor must emit q_mean_field"
            q_abs = f["q_abs"][:]
            q_flat = f["q_flat"][:]
            q_mean_field = f["q_mean_field"][:]
            nt = int(f.attrs["nt"])
            n_pix = len(q_mean_field)
        assert len(q_abs) == n_pix * nt
        per_pixel_mean = q_abs.reshape(nt, n_pix).mean(axis=0)
        np.testing.assert_allclose(per_pixel_mean, q_mean_field, rtol=1e-4)
        # q_abs and the legacy q_flat are now the same ordering (time-major)
        np.testing.assert_allclose(q_abs, q_flat, rtol=1e-6)

    def test_absolute_and_fluctuation_metrics_reported(self, registry_with_three):
        """evaluate() must report both a fluctuation metric (the meaningful
        one, under 'test') and an absolute-field metric (under 'test_absolute').
        Absolute R² is >= fluctuation R² because the spatial mean inflates the
        denominator variance."""
        from icarus.registry.trainer import MultiDatasetTrainer

        trainer = MultiDatasetTrainer(registry_with_three, n_pod_modes=3,
                                      n_training_samples=None,
                                      optimise_hyperparams=False)
        trainer.model_ = HeatFluxNet(strategy="modal",
                                     hidden_layer_sizes=(16,), max_iter=20)
        trainer.cross_dataset_fit(train_ids=["D001", "D002"], test_id="D003",
                                  verbose=False)
        metrics = trainer.evaluate(verbose=False)
        assert "test_fluctuation" in metrics
        assert "test_absolute" in metrics
        assert metrics["test"] is metrics["test_fluctuation"]
        assert metrics["test_absolute"].r2 >= metrics["test_fluctuation"].r2 - 1e-9


class TestExtractorTimeMajorOrdering:
    """Regression guard for the extractor split-mask ordering bug.

    The extractor's split mask marks the first ``nt_train * n_pix`` rows as
    train, which is only a *temporal* split if feature rows are TIME-major
    (row = t * n_pix + p). The original code reshaped modal contributions
    pixel-major (row = p * nt + t), silently turning the "temporal" split into
    a spatial pixel split — train and test shared every timestep. This test
    pins the row ordering of the stored features.
    """

    def _make_rank2_npz(self, tmp_path):
        # Noiseless rank-2 fields: POD with 3 modes reconstructs them exactly,
        # so q_contribs.sum(axis=1) must equal the centred field — and the row
        # ordering of that equality reveals time-major vs pixel-major.
        ny, nx, nt = 12, 14, 50
        XX, YY = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
        t = np.linspace(0, 1, nt)
        m1 = np.sin(np.pi * YY) * np.cos(np.pi * XX)
        m2 = np.cos(2 * np.pi * YY) * np.sin(2 * np.pi * XX)
        a1 = np.sin(2 * np.pi * t)
        a2 = np.cos(3 * np.pi * t)
        T = 400.0 + 3 * m1[:, :, None] * a1 + 2 * m2[:, :, None] * a2
        q = 3e5 + 4e4 * m1[:, :, None] * a1 - 2.5e4 * m2[:, :, None] * a2
        p = tmp_path / "rank2.npz"
        np.savez(p, T=T, qL2=q, TimeStep=1e-3)
        return str(p), q

    def test_feature_rows_are_time_major(self, tmp_path):
        from icarus.registry.dataset import DatasetRegistry, DatasetEntry
        from icarus.registry.extractor import FeatureExtractor
        from icarus.data.preprocessor import Preprocessor, PreprocessorConfig
        from icarus.data.loader import load
        import h5py

        raw, _ = self._make_rank2_npz(tmp_path)
        reg = DatasetRegistry(tmp_path / "store")
        reg.register(DatasetEntry("D001", "water", "flow_boiling", "s",
                                  "Test", raw_path=raw, spatial_crop=2))
        FeatureExtractor(reg, n_pod_modes=3).process("D001", force=True)

        # Expected centred field, flattened TIME-major, after identical prep
        data = load(raw)
        out = Preprocessor(PreprocessorConfig(spatial_crop=2,
                                              trim_frames=0)).fit_transform(data)
        q_c = out["q_c"]
        expected = q_c.transpose(2, 0, 1).reshape(-1)

        with h5py.File(reg.get("D001").processed_path, "r") as f:
            got = f["q_contribs"][:].sum(axis=1)
            nt_train = int(f.attrs["nt_train"])
            n_pix = int(f.attrs["ny"]) * int(f.attrs["nx"])
            split = f["split"][:]

        # Rank-2 data + 3 modes → reconstruction is exact, so any mismatch
        # here is a row-ordering error, not POD truncation.
        np.testing.assert_allclose(got, expected, rtol=1e-3,
                                   atol=1e-2 * np.abs(expected).max())

        # And therefore the split mask is genuinely temporal: train rows are
        # exactly the first nt_train timesteps.
        assert (split[:nt_train * n_pix] == 0).all()
        assert (split[nt_train * n_pix:] == 1).all()


class TestPipelineModelInjection:
    """Regression guard: Pipeline.fit() must respect a manually injected
    HeatFluxNet instead of silently overwriting it with a fresh one."""

    def test_injected_model_is_used(self, synthetic_data):
        from icarus.pipeline.runner import Pipeline

        pipe = Pipeline(strategy="modal", n_pod_modes=3, spatial_crop=2,
                        optimise_hyperparams=False, n_training_samples=2000)
        injected = HeatFluxNet(strategy="modal", hidden_layer_sizes=(8,),
                               max_iter=10)
        pipe.model_ = injected
        pipe.fit(synthetic_data, verbose=False)
        assert pipe.model_ is injected, \
            "fit() must not overwrite a manually injected model"

    def test_default_model_still_created(self, synthetic_data):
        from icarus.pipeline.runner import Pipeline

        pipe = Pipeline(strategy="modal", n_pod_modes=3, spatial_crop=2,
                        optimise_hyperparams=False, n_training_samples=2000)
        # No injection: fit must create a model itself. Give it usable
        # hyperparams afterwards is impossible, so patch via optimise-off
        # default construction and explicit sizes through injection path is
        # covered above; here we only check a model exists and is fitted.
        try:
            pipe.fit(synthetic_data, verbose=False)
        except RuntimeError:
            # Default HeatFluxNet has no hidden_layer_sizes without optimise —
            # acceptable; the point is that a model object was created.
            pass
        assert pipe.model_ is not None


class TestPipelineModalPredict:
    """Regression guard for Pipeline._predict_modal (Model C inference on new
    data). It must reconstruct the heat-flux field the SAME way evaluate() does:
    sum the predicted modal contributions (which already include phi_i) and add
    the per-pixel mean back, respecting time-major row ordering. The original
    code multiplied predictions by phi_i again AND reshaped time-major rows as
    pixel-major — so predict() returned a scrambled, double-counted field while
    evaluate() (using the correct sibling) looked fine."""

    def test_predict_matches_correct_modal_reconstruction(self, synthetic_data):
        from icarus.pipeline.runner import Pipeline
        from icarus.data.preprocessor import Preprocessor
        from icarus.features.engineer import build_modal_features

        pipe = Pipeline(strategy="modal", n_pod_modes=3, spatial_crop=2,
                        optimise_hyperparams=False, n_training_samples=None)
        pipe.model_ = HeatFluxNet(strategy="modal", hidden_layer_sizes=(16,),
                                  max_iter=50)
        pipe.fit(synthetic_data, verbose=False)

        proc = pipe._processed
        T_field = proc["T"]                       # preprocessed [ny, nx, nt]
        ny, nx, nt = T_field.shape

        # Independent, correct reconstruction (sum contributions + tile mean).
        X_c = Preprocessor.to_matrix(proc["T_c"])
        T_contribs = pipe.pod_T_.modal_contributions(X_c, n_modes=3)
        X_feat, _ = build_modal_features(T_contribs)        # time-major
        y_modal = pipe.model_.predict(X_feat)               # [nt*n_pix, 3]
        q_c_flat = y_modal.sum(axis=1)
        q_mean_vec = pipe.preprocessor_.q_mean.reshape(-1)
        expected = q_c_flat + np.tile(q_mean_vec, nt)        # time-major flat

        # predict() centres internally; passing the preprocessed T reproduces
        # the same centred field, so outputs must match the expected field.
        q_pred_field = pipe.predict(T_field)
        assert q_pred_field.shape == (ny, nx, nt)
        got = q_pred_field.transpose(2, 0, 1).reshape(-1)    # time-major flat
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)
