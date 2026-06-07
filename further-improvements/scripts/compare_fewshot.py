"""Paired few-shot comparison: NEW transfer mechanism vs the original mt_cotrain.

For every (endpoint, arm, n) cell shared by the new sweep and results-fullexp, this pairs the two
arms on (seed) and asks a single question: does the new mechanism LOWER RAE? It reports the mean
RAE of each, the paired delta (old - new; positive == improvement), a seed-level paired bootstrap
95% CI on that delta, and the seed sign-consistency (how many of the 5 seeds improved).

Light CPU only (reads small per-molecule parquets + the fullexp results.parquet, recomputes RAE).
Run inside the project container; the heavy prediction *generation* already happened on GPU.

    PYTHONPATH=further-improvements/src python further-improvements/scripts/compare_fewshot.py \
        --new-preds further-improvements/results-fewshot/pf/preds --label pretrain_finetune \
        --old-results 'results-fullexp/*/results.parquet' --old-mechanism mt_cotrain \
        --out further-improvements/outputs/compare_pretrain_finetune.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import rae

ARMS = ["external", "both"]


def _new_per_seed(preds_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(preds_dir / "*.parquet"))):
        if p.endswith(".meta.parquet"):
            continue
        meta_path = p.replace(".parquet", ".meta.parquet")
        if not Path(meta_path).exists():
            continue
        preds = pd.read_parquet(p)
        meta = pd.read_parquet(meta_path).iloc[0].to_dict()
        y_true = preds["y_true"].to_numpy(dtype=float)
        y_pred = preds["y_pred"].to_numpy(dtype=float)
        test_range = float(meta.get("test_range") or (np.nanmax(y_true) - np.nanmin(y_true)))
        rows.append({
            "endpoint": meta.get("endpoint"),
            "arm": meta.get("arm"),
            "n": meta.get("n"),
            "seed": meta.get("seed"),
            "rae_new": rae(y_true, y_pred, definition="range_normalized", test_range=test_range),
        })
    df = pd.DataFrame(rows)
    df["n"] = df["n"].astype("Int64")
    return df


def _old_per_seed(results_glob: str, mechanism: str) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in glob.glob(results_glob)]
    if not frames:
        raise SystemExit(f"no fullexp results matched {results_glob!r}")
    old = pd.concat(frames, ignore_index=True)
    old = old[(old["mechanism"] == mechanism) & (old["arm"].isin(ARMS))].copy()
    old = old.rename(columns={"rae": "rae_old"})
    old["n"] = old["n"].astype("Int64")
    return old[["endpoint", "arm", "n", "seed", "rae_old"]]


def _paired_bootstrap(delta: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired new-vs-old few-shot RAE comparison.")
    ap.add_argument("--new-preds", type=Path, required=True)
    ap.add_argument("--label", required=True, help="name of the new mechanism, e.g. pretrain_finetune")
    ap.add_argument("--old-results", default="results-fullexp/*/results.parquet")
    ap.add_argument("--old-mechanism", default="mt_cotrain")
    ap.add_argument("--out", type=Path, default=Path("further-improvements/outputs/compare.csv"))
    args = ap.parse_args()

    new = _new_per_seed(args.new_preds)
    old = _old_per_seed(args.old_results, args.old_mechanism)
    merged = new.merge(old, on=["endpoint", "arm", "n", "seed"], how="inner")
    if merged.empty:
        raise SystemExit("no overlapping (endpoint,arm,n,seed) cells — check the sweep finished")

    out_rows = []
    for (ep, arm, n), g in merged.groupby(["endpoint", "arm", "n"]):
        delta = (g["rae_old"] - g["rae_new"]).to_numpy(dtype=float)  # >0 == new is better
        lo, hi = _paired_bootstrap(delta)
        out_rows.append({
            "endpoint": ep, "arm": arm, "n": int(n), "seeds": len(g),
            "rae_old": g["rae_old"].mean(), "rae_new": g["rae_new"].mean(),
            "delta_mean": delta.mean(), "ci95_lo": lo, "ci95_hi": hi,
            "improved_seeds": int((delta > 0).sum()),
            "significant": "YES" if lo > 0 else ("WORSE" if hi < 0 else "ns"),
        })
    res = pd.DataFrame(out_rows).sort_values(["endpoint", "arm", "n"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n=== {args.label}  vs  {args.old_mechanism}  (delta = old - new; >0 means LOWER error) ===")
    print(res.to_string(index=False))
    wins = res[res["significant"] == "YES"]
    print(f"\nsignificant improvements (CI excludes 0): {len(wins)}/{len(res)} cells")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
