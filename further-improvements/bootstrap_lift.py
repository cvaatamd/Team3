"""Paired bootstrap CI on transfer-arm lift over baseline, per endpoint.

Turns "the mean looks lower" into a confidence interval + sign check, reading only the per-seed
scalars already stored in results.parquet — no retraining.

Usage (from project root, inside the env):
    python further-improvements/bootstrap_lift.py

Honest caveat: this bootstraps over the 5 seeds, so the CI reflects seed/subsample variance, not
test-molecule variance. Once per-molecule predictions are persisted (see further-improvements/
patches.md, item 2), re-run the same logic at the molecule level for much tighter CIs.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"rae", "mae", "rmse"}


def load_results(pattern: str = "results-fullexp/*/results.parquet") -> pd.DataFrame:
    """Full-exp results are split per round; concat them into one frame."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet matched {pattern!r}; run from the project root")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # collected tables use `endpoint`; raw EvalResult dumps use `target_endpoint`
    if "endpoint" not in df.columns and "target_endpoint" in df.columns:
        df = df.rename(columns={"target_endpoint": "endpoint"})
    return df


def paired_deltas(
    df: pd.DataFrame,
    endpoint: str,
    arm: str,
    metric: str = "rae",
    n: int | None = None,
    baseline: str = "baseline",
) -> np.ndarray:
    """Per-seed improvement of `arm` over `baseline`. Positive == better, sign-corrected."""
    sub = df[df["endpoint"] == endpoint]
    sub = sub[sub["n"].isna()] if n is None else sub[sub["n"] == n]  # n=full stored as null
    b = sub[sub["arm"] == baseline].set_index("seed")[metric]
    a = sub[sub["arm"] == arm].set_index("seed")[metric]
    seeds = b.index.intersection(a.index)
    if len(seeds) == 0:
        return np.array([])
    if metric in LOWER_IS_BETTER:
        return (b.loc[seeds] - a.loc[seeds]).to_numpy()
    return (a.loc[seeds] - b.loc[seeds]).to_numpy()


def boot_ci(d: np.ndarray, n_boot: int = 10_000, ci: float = 95.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(d.mean()), float(lo), float(hi)


def main() -> None:
    df = load_results()
    checks = [
        ("MBPB", "external"), ("KSOL", "external"), ("MPPB", "external"),
        ("MBPB", "both"), ("MPPB", "both"),
    ]
    print(f"{'endpoint':8s} {'arm':8s} {'metric':8s}  Δ (lift)   95% CI                signif   "
          f"sign-consistency")
    print("-" * 88)
    for ep, arm in checks:
        for metric in ("rae", "spearman"):
            d = paired_deltas(df, ep, arm, metric)
            if len(d) == 0:
                continue
            m, lo, hi = boot_ci(d)
            sig = "SIGNIFICANT" if lo > 0 else "n.s."
            wins = int((d > 0).sum())
            print(f"{ep:8s} {arm:8s} {metric:8s}  {m:+.4f}   [{lo:+.4f}, {hi:+.4f}]   "
                  f"{sig:11s}  {wins}/{len(d)} seeds")


if __name__ == "__main__":
    main()
