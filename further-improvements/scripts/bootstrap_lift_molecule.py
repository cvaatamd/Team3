"""Molecule-level paired bootstrap CI on transfer lift over baseline.

The seed-level version (``further-improvements/bootstrap_lift.py``) bootstraps over 5 seeds, so its
CIs are wide. This version bootstraps over the ~2.3k held-out TEST MOLECULES — the fixed test set
shared by every arm/seed — giving much tighter, properly-paired intervals.

It requires the per-molecule predictions persisted by Patch 1, so run it only AFTER the regen
sbatch has produced ``<results_dir>/preds/``. Reading parquets + resampling is light CPU; the
heavy model training happens on GPU via the regen job, never on a login node.

Method:
  * For each (endpoint, arm) at n=full, average y_pred across the 5 seeds (a small ensemble that
    removes seed/init noise), keeping the shared test y_true.
  * Bootstrap test-molecule indices with replacement; recompute the metric for baseline and arm
    on each resample and take the paired delta (sign-corrected so positive == improvement).
  * Report mean lift, 95% percentile CI, and the fraction of resamples favouring the arm.

Usage (inside the project container):
    python further-improvements/scripts/bootstrap_lift_molecule.py \
        --preds-dir results-fullexp-regen/mbpb/preds \
        --preds-dir results-fullexp-regen/ksol/preds \
        --preds-dir results-fullexp-regen/mppb/preds
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"rae", "mae", "rmse"}


def _metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str, test_range: float) -> float:
    err = np.abs(y_true - y_pred)
    if metric == "mae":
        return float(err.mean())
    if metric == "rmse":
        return float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    if metric == "rae":
        return float(err.mean() / test_range) if test_range > 0 else float("nan")
    if metric == "spearman":
        from scipy.stats import spearmanr
        return float(spearmanr(y_true, y_pred).correlation)
    raise ValueError(metric)


def _load_cell_preds(preds_dir: Path, n_tag: str = "full"):
    """Return {(endpoint, arm): (y_true, mean_y_pred_over_seeds, test_range)} for the given n."""
    by_cell: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    truth: dict[tuple[str, str], np.ndarray] = {}
    ranges: dict[tuple[str, str], float] = {}
    for p in sorted(glob.glob(str(preds_dir / "*.parquet"))):
        if p.endswith(".meta.parquet"):
            continue
        meta = pd.read_parquet(p.replace(".parquet", ".meta.parquet")).iloc[0].to_dict()
        if str(meta.get("n")) != n_tag:
            continue
        key = (meta["endpoint"], meta["arm"])
        preds = pd.read_parquet(p)
        by_cell[key].append(preds["y_pred"].to_numpy(dtype=float))
        truth[key] = preds["y_true"].to_numpy(dtype=float)
        yt = truth[key]
        ranges[key] = float(meta.get("test_range") or (np.nanmax(yt) - np.nanmin(yt)))
    out = {}
    for key, preds_list in by_cell.items():
        out[key] = (truth[key], np.mean(np.vstack(preds_list), axis=0), ranges[key])
    return out


def bootstrap_delta(y_true, base_pred, arm_pred, metric, test_range,
                    n_boot=10_000, seed=0, ci=95.0):
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(y_true) | np.isnan(base_pred) | np.isnan(arm_pred))
    yt, bp, ap = y_true[mask], base_pred[mask], arm_pred[mask]
    m = len(yt)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, m, m)
        mb = _metric(yt[idx], bp[idx], metric, test_range)
        ma = _metric(yt[idx], ap[idx], metric, test_range)
        deltas[i] = (mb - ma) if metric in LOWER_IS_BETTER else (ma - mb)
    point = ((_metric(yt, bp, metric, test_range) - _metric(yt, ap, metric, test_range))
             if metric in LOWER_IS_BETTER
             else (_metric(yt, ap, metric, test_range) - _metric(yt, bp, metric, test_range)))
    lo, hi = np.percentile(deltas, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(point), float(lo), float(hi), float((deltas > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", type=Path, action="append", required=True)
    ap.add_argument("--n-tag", default="full", help="sweep size to analyze (default: full)")
    ap.add_argument("--arms", nargs="+", default=["external", "both", "intra_task"])
    ap.add_argument("--metrics", nargs="+", default=["rae", "spearman"])
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    cells: dict = {}
    for d in args.preds_dir:
        cells.update(_load_cell_preds(d, n_tag=args.n_tag))
    if not cells:
        raise SystemExit("no prediction parquets found — run the regen sbatch first")

    endpoints = sorted({ep for (ep, _arm) in cells})
    print(f"{'endpoint':28s} {'arm':10s} {'metric':9s}  lift      95% CI                P(arm>base)")
    print("-" * 92)
    for ep in endpoints:
        base = cells.get((ep, "baseline"))
        if base is None:
            print(f"{ep:28s}  (no baseline preds — skipping)")
            continue
        y_true, base_pred, test_range = base
        for arm in args.arms:
            cell = cells.get((ep, arm))
            if cell is None:
                continue
            _yt, arm_pred, _rng = cell
            for metric in args.metrics:
                pt, lo, hi, p = bootstrap_delta(
                    y_true, base_pred, arm_pred, metric, test_range, n_boot=args.n_boot)
                sig = "SIGNIFICANT" if lo > 0 else "n.s."
                print(f"{ep:28s} {arm:10s} {metric:9s}  {pt:+.4f}  "
                      f"[{lo:+.4f}, {hi:+.4f}]  {p:5.1%}  {sig}")


if __name__ == "__main__":
    main()
