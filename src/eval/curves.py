"""Learning-curve plotting for the n-vs-RAE sweep (§4)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse (endpoint, arm, n, seed) -> mean / std over seeds."""
    g = results.groupby(["endpoint", "arm", "n"], dropna=False)["rae"]
    out = g.agg(["mean", "std", "count"]).reset_index().rename(
        columns={"mean": "rae_mean", "std": "rae_std", "count": "n_seeds"}
    )
    return out


def plot_curves(results: pd.DataFrame, out_path: Path) -> Path:
    """One subplot per endpoint; arms as lines; shaded ±1 std band over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate_results(results)
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
            ax.plot(s["n"], s["rae_mean"], marker="o", label=arm)
            if s["rae_std"].notna().any():
                ax.fill_between(
                    s["n"], s["rae_mean"] - s["rae_std"], s["rae_mean"] + s["rae_std"], alpha=0.15
                )
        ax.set_title(ep)
        ax.set_xlabel("n target labels")
        ax.set_ylabel("RAE")
        ax.set_xscale("log")
        ax.legend(fontsize=8)

    for j in range(n_eps, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ma_rae(results: pd.DataFrame, out_path: Path) -> Path:
    """Aggregate MA-RAE curve across endpoints (§Primary deliverables)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # First average over seeds, then macro-avg over endpoints.
    per_seed = results.groupby(["arm", "n", "seed"])["rae"].mean().reset_index()
    agg = per_seed.groupby(["arm", "n"])["rae"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in agg["arm"].unique():
        s = agg[agg["arm"] == arm].sort_values("n")
        ax.plot(s["n"], s["mean"], marker="o", label=arm)
        if s["std"].notna().any():
            ax.fill_between(s["n"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
    ax.set_title("Macro-averaged RAE (MA-RAE) vs n")
    ax.set_xlabel("n target labels")
    ax.set_ylabel("MA-RAE")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
