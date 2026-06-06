"""Learning-curve plotting for the n-vs-RAE sweep (§4)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Metrics where smaller is better (used for axis labelling / direction-aware reports).
LOWER_IS_BETTER = {"rae", "mae", "rmse"}


def metric_label(metric: str) -> str:
    """Human-readable axis label, e.g. 'RAE (lower is better)'."""
    direction = "lower" if metric in LOWER_IS_BETTER else "higher"
    return f"{metric.upper()} ({direction} is better)"


def normalize_endpoint_column(results: pd.DataFrame) -> pd.DataFrame:
    """Accept either `target_endpoint` (EvalResult) or `endpoint` (collected table)."""
    if results.empty or "endpoint" in results.columns:
        return results
    if "target_endpoint" in results.columns:
        return results.rename(columns={"target_endpoint": "endpoint"})
    return results


def aggregate_results(results: pd.DataFrame, metric: str = "rae") -> pd.DataFrame:
    """Collapse (endpoint, arm, n, seed) -> mean / std over seeds for `metric`."""
    results = normalize_endpoint_column(results)
    if metric not in results.columns:
        raise KeyError(f"metric {metric!r} not in results columns: {list(results.columns)}")
    g = results.groupby(["endpoint", "arm", "n"], dropna=False)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index().rename(
        columns={"count": "n_seeds"}
    )
    return out


def plot_curves(results: pd.DataFrame, out_path: Path, metric: str = "rae") -> Path:
    """One subplot per endpoint; arms as lines; shaded ±1 std band over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate_results(results, metric=metric)
    endpoints = agg["endpoint"].unique().tolist()
    arms = agg["arm"].unique().tolist()
    n_eps = len(endpoints)
    cols = min(3, n_eps)
    rows = (n_eps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, ep in enumerate(endpoints):
        ax = axes[i // cols][i % cols]
        sub = agg[agg["endpoint"] == ep]
        for arm in arms:
            s = sub[sub["arm"] == arm].sort_values("n")
            if s.empty:
                continue
            ax.plot(s["n"], s["mean"], marker="o", label=arm)
            if s["std"].notna().any():
                ax.fill_between(
                    s["n"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15
                )
        ax.set_title(ep)
        ax.set_xlabel("n target labels")
        ax.set_ylabel(metric_label(metric))
        ax.set_xscale("log")
        ax.legend(fontsize=8)

    for j in range(n_eps, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ma_rae(results: pd.DataFrame, out_path: Path, metric: str = "rae") -> Path:
    """Macro-averaged metric curve across endpoints (§Primary deliverables)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = normalize_endpoint_column(results)
    if metric not in results.columns:
        raise KeyError(f"metric {metric!r} not in results columns: {list(results.columns)}")
    # First average over seeds, then macro-avg over endpoints.
    per_seed = results.groupby(["arm", "n", "seed"])[metric].mean().reset_index()
    agg = per_seed.groupby(["arm", "n"])[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in agg["arm"].unique():
        s = agg[agg["arm"] == arm].sort_values("n")
        ax.plot(s["n"], s["mean"], marker="o", label=arm)
        if s["std"].notna().any():
            ax.fill_between(s["n"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
    ax.set_title(f"Macro-averaged {metric.upper()} vs n")
    ax.set_xlabel("n target labels")
    ax.set_ylabel(f"MA-{metric.upper()}")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
