import math

import numpy as np

from eval.metrics import diagnostics, ma_rae, rae


def test_rae_range_normalized_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rae(y, y, definition="range_normalized") == 0.0


def test_rae_range_normalized_value():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 9.0])
    # MAE = 1.0; range = 10.0; RAE = 0.1
    assert math.isclose(rae(y_true, y_pred, definition="range_normalized"), 0.1)


def test_rae_vs_mean_baseline_is_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    y_pred = np.full_like(y, y.mean())
    assert math.isclose(rae(y, y_pred, definition="vs_mean"), 1.0, abs_tol=1e-9)


def test_rae_handles_nan_in_pred():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, np.nan, 2.5])
    # masked: MAE on the two valid pairs = (0.5 + 0.5)/2 = 0.5; range = 2.0 → 0.25
    assert math.isclose(rae(y_true, y_pred), 0.25)


def test_ma_rae_ignores_nan_endpoints():
    per = {"A": 0.1, "B": 0.3, "C": float("nan")}
    assert math.isclose(ma_rae(per), 0.2)


def test_diagnostics_bundle_shapes():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    yhat = y + rng.normal(scale=0.5, size=100)
    d = diagnostics(y, yhat)
    assert d.n == 100
    assert 0.0 <= d.rae <= 1.0
    assert d.mae > 0
    assert d.rmse >= d.mae
