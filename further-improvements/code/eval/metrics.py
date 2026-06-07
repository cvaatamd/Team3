"""Pluggable RAE / MA-RAE plus diagnostics (§1).

The RAE definition is intentionally pluggable. The official scoring code on the challenge
HuggingFace Space is the source of truth — wire it up via `rae_official` once verified.

Two definitions implemented:
- `range_normalized`: per-endpoint MAE divided by the test set's dynamic range (max - min)
- `vs_mean`: per-endpoint MAE divided by the MAE of a mean predictor (i.e. MAE(y, mean(y)))

`MA_RAE` is the macro-average across endpoints — the challenge's headline metric.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger(__name__)

RaeDef = Literal["range_normalized", "vs_mean", "official"]


def rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    definition: RaeDef = "range_normalized",
    test_range: float | None = None,
) -> float:
    """Compute RAE for one endpoint, on the *native* scale (caller inverts transform first)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    mae = mean_absolute_error(y_true, y_pred)

    if definition == "range_normalized":
        rng = test_range if test_range is not None else float(y_true.max() - y_true.min())
        return float(mae / rng) if rng > 0 else float("nan")
    if definition == "vs_mean":
        denom = mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))
        return float(mae / denom) if denom > 0 else float("nan")
    if definition == "official":
        # Patch 2 (further-improvements): the OpenADMET-ExpansionRx challenge scores per-endpoint
        # MAE normalized by that endpoint's dynamic range on the *native-scale* test labels, then
        # macro-averages across endpoints (MA-RAE). That is identical to `range_normalized` with
        # the native-scale test range, which is what the pipeline already passes in.
        #
        # >>> VERIFY-THEN-LOCK: before quoting "official" numbers, diff this against the scoring
        #     code in the HF Space `openadmet/OpenADMET-ExpansionRx-Challenge`. Confirm (a) native
        #     vs log space, (b) min-max vs a robust/percentile range, (c) any fixed published
        #     normalizer. If (a)-(c) all match min-max native range, this alias is exact; otherwise
        #     replace `rng` below with the official constant/range and keep the same call site.
        rng = test_range if test_range is not None else float(y_true.max() - y_true.min())
        return float(mae / rng) if rng > 0 else float("nan")
    raise ValueError(f"unknown RAE definition: {definition}")


def ma_rae(
    per_endpoint: dict[str, float],
) -> float:
    """Macro-average RAE across endpoints (ignoring NaN endpoints)."""
    vals = [v for v in per_endpoint.values() if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class MetricBundle:
    rae: float
    mae: float
    rmse: float
    r2: float
    spearman: float
    n: int

    def to_dict(self) -> dict:
        return {
            "rae": self.rae, "mae": self.mae, "rmse": self.rmse,
            "r2": self.r2, "spearman": self.spearman, "n": self.n,
        }


def diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    rae_definition: RaeDef = "range_normalized",
    test_range: float | None = None,
) -> MetricBundle:
    """All standard metrics in one shot — keep RAE/MAE/RMSE/R²/Spearman alongside (§1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        nan = float("nan")
        return MetricBundle(rae=nan, mae=nan, rmse=nan, r2=nan, spearman=nan, n=0)
    rho = spearmanr(y_true, y_pred).correlation if len(y_true) > 1 else float("nan")
    return MetricBundle(
        rae=rae(y_true, y_pred, definition=rae_definition, test_range=test_range),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        spearman=float(rho if rho is not None else float("nan")),
        n=int(len(y_true)),
    )


# Plug points: the planner can swap `RAE_FN` once the official definition is verified.
RAE_FN: Callable[..., float] = rae
