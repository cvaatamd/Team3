"""Offline re-score: recompute RAE under any definition from persisted predictions.

Consumes the per-run prediction parquets written by Patch 1 (``<results_dir>/preds/*.parquet``)
and recomputes RAE for every definition WITHOUT retraining. Confirms the headline numbers hold
and that ``official`` == ``range_normalized`` (the EXPERIMENT.md claim).

This is light CPU (reads small parquets, no model). It still needs the patched code on PYTHONPATH
because it calls ``eval.metrics.rae``:

    PYTHONPATH=further-improvements/src \
      python further-improvements/scripts/rescore_official.py \
        --preds-dir results-fullexp-regen/mbpb/preds

Run it inside the project container (which has pandas/pyarrow); do NOT run heavy jobs on a login
node — re-scoring is cheap, but prediction *generation* happens on GPU via the regen sbatch.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import rae  # patched copy: implements the `official` branch

DEFINITIONS = ["range_normalized", "vs_mean", "official"]


def _load_pairs(preds_dir: Path) -> list[dict]:
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
        row = {k: meta.get(k) for k in ("endpoint", "arm", "n", "seed", "mechanism")}
        for d in DEFINITIONS:
            row[f"rae_{d}"] = rae(y_true, y_pred, definition=d, test_range=test_range)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score RAE definitions from persisted preds.")
    ap.add_argument("--preds-dir", type=Path, action="append", required=True,
                    help="a <results_dir>/preds folder; repeat for multiple rounds")
    ap.add_argument("--out", type=Path, default=Path("further-improvements/rescore_official.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    for d in args.preds_dir:
        rows.extend(_load_pairs(d))
    if not rows:
        raise SystemExit("no prediction parquets found — run the regen sbatch first")

    df = pd.DataFrame(rows)
    agg = (df.groupby(["endpoint", "arm", "n"], dropna=False)[[f"rae_{d}" for d in DEFINITIONS]]
             .mean().reset_index())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)

    # Headline check: does `official` agree with `range_normalized`?
    delta = (agg["rae_official"] - agg["rae_range_normalized"]).abs()
    print(f"wrote {args.out}  ({len(agg)} cells)")
    print(f"max |official - range_normalized| over all cells: {delta.max():.2e} "
          f"({'AGREE' if delta.max() < 1e-9 else 'DIFFER — inspect the official spec'})")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
