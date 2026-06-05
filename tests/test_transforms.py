import numpy as np
import pandas as pd

from data.expansionrx import EndpointSpec, _forward, apply_transforms, invert_transform


def test_log10_roundtrip():
    spec = EndpointSpec(column="X", transform="log10", floor=1e-3)
    raw = np.array([0.001, 1.0, 100.0, 1e4])
    fwd = _forward(raw, spec)
    back = invert_transform(fwd, spec)
    np.testing.assert_allclose(back, raw, rtol=1e-9)


def test_log10_floors_zero():
    spec = EndpointSpec(column="X", transform="log10", floor=1e-3)
    out = _forward(np.array([0.0, -1.0, 1.0]), spec)
    # floored to 1e-3 -> log10(1e-3) = -3
    assert out[0] == -3.0
    assert out[1] == -3.0
    assert out[2] == 0.0


def test_logit_pct_roundtrip():
    spec = EndpointSpec(column="X", transform="logit_pct", clip=(1e-3, 1 - 1e-3))
    raw = np.array([10.0, 50.0, 90.0])  # % unbound
    fwd = _forward(raw, spec)
    back = invert_transform(fwd, spec)
    np.testing.assert_allclose(back, raw, rtol=1e-6)


def test_apply_transforms_adds_columns():
    spec = EndpointSpec(column="HLM CLint", transform="log10", floor=1e-3)
    df = pd.DataFrame({"HLM CLint": [1.0, 10.0, 100.0]})
    apply_transforms(df, [spec])
    assert "HLM CLint__t" in df.columns
    np.testing.assert_allclose(df["HLM CLint__t"].values, [0.0, 1.0, 2.0])
