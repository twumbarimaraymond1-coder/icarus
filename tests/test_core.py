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

    def test_q_abs_is_pixel_major_aligned(self, registry_with_three):
        """The extractor must store q_abs in PIXEL-major order, row-aligned
        with q_contribs (row = pixel*nt + t). Guards the absolute-R² fix:
        reshaping q_abs to [n_pix, nt] and meaning over time must equal the
        separately stored per-pixel mean field."""
        import h5py
        reg = registry_with_three
        path = reg.get("D003").processed_path
        with h5py.File(path, "r") as f:
            assert "q_abs" in f, "extractor must emit q_abs"
            assert "q_mean_field" in f, "extractor must emit q_mean_field"
            q_abs = f["q_abs"][:]
            q_mean_field = f["q_mean_field"][:]
            nt = int(f.attrs["nt"])
            n_pix = len(q_mean_field)
        assert len(q_abs) == n_pix * nt
        per_pixel_mean = q_abs.reshape(n_pix, nt).mean(axis=1)
        np.testing.assert_allclose(per_pixel_mean, q_mean_field, rtol=1e-4)

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
