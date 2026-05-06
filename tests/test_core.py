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
        T, q = out["T"], out["q"]
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
