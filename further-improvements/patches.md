# Exact changes — prediction persistence, official RAE, negative control

Three small, surgical changes. Line references are against the repo as of this writeup; adjust if
the files have moved. Nothing here is applied automatically — these are specs to drop in.

---

## Patch 1 — persist per-molecule predictions (unblocks everything)

**Why:** `src/experiment/runner.py::run_one` stores only the `EvalResult` scalars, so re-scoring
with a new RAE definition (or a molecule-level bootstrap) is impossible without re-training all
600 jobs. Save `(y_true, y_pred)` once and both become free offline operations.

**Where:** `src/agents/ml_agent.py::MLAgent.run`, right after the metrics are computed
(around lines 64–87). Return the arrays alongside the result, or write a sidecar parquet.

Minimal sidecar approach (no contract change) — add to `MLAgent.run` just before `return`:

```python
# --- persist predictions for free re-scoring (official RAE, molecule-level bootstrap) ---
preds_dir = getattr(spec, "preds_dir", None)
if preds_dir is not None:
    import pandas as pd
    from pathlib import Path
    Path(preds_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_parquet(
        Path(preds_dir) / f"{spec.pool.target_endpoint}__{spec.arm}__n{spec.n}__s{spec.seed}.parquet"
    )
```

Then thread a `preds_dir` (e.g. `results_dir / "preds"`) through `TrainSpec` and
`runner.run_one` / `planner.in_process_dispatcher`. ~5 lines total.

A re-score script can then load each preds parquet + the endpoint's native test range and call
`eval.metrics.rae(..., definition="official")` — no GPU, no retrain.

---

## Patch 2 — wire the official RAE

**Where:** `src/eval/metrics.py::rae`, the `official` branch (currently raises
`NotImplementedError`).

**First, verify against the source of truth** — the challenge HF Space scoring code
(`openadmet/OpenADMET-ExpansionRx-Challenge`). Confirm three things, because they change the number:
1. **Space:** native scale vs log/transformed space (we currently invert to native — line ~65 of
   `ml_agent.py`).
2. **Denominator:** `max - min` vs a robust/percentile range vs a fixed published constant.
3. **Aggregation:** per-endpoint normalize then macro-average (our `ma_rae`).

If the official denominator is the native min–max test range, our `range_normalized` is **already
the official metric** and this patch is just an alias that locks it in:

```python
    if definition == "official":
        # Verified equal to range_normalized against the HF Space scoring code on YYYY-MM-DD.
        rng = test_range if test_range is not None else float(y_true.max() - y_true.min())
        return float(mae / rng) if rng > 0 else float("nan")
```

If it differs (e.g. fixed published ranges), replace the `rng` line with the official constants.
Either way, re-score from the Patch-1 predictions and report both `range_normalized` and `official`
so reviewers can see they agree (or by how much they differ).

---

## Patch 3 — shuffled-external negative control

**Why:** proves the external lift is signal, not just an extra regularizing column.

**Cheapest form (no schema change):** add a permute toggle to
`src/data/harmonize.py::harmonize_external`. After the source aux column is built and standardized
(around line 103), permute the labels under a fixed seed:

```python
def harmonize_external(target_df, target_spec, source, mapping, *,
                       force_strategy=None, standardize=True, shuffle_seed=None):
    ...
    src = src[["SMILES", aux_col]].dropna()
    if shuffle_seed is not None:
        # NEGATIVE CONTROL: keep the marginal distribution, destroy the structure-activity link.
        src[aux_col] = src[aux_col].sample(frac=1.0, random_state=shuffle_seed).to_numpy()
    if standardize:
        ...
```

Thread `shuffle_seed` up through `DataAgent.build_pool` (via a `PoolRequest` field, default `None`)
and run the `external` / `both` arms once with it set, into a separate results dir
(e.g. `results-fullexp-shuffled/`). Expected outcome: the shuffled external arm collapses toward
baseline; the real external arm beats it clearly. That gap is the proof.

**Deliverable:** one extra table in EXPERIMENT.md — "real external vs shuffled external (Δ RAE)" —
per endpoint. It directly answers the skeptic's "any extra column would help."

---

## Suggested order of operations
1. Patch 1 (persist preds) — unblocks 2 and the molecule-level bootstrap.
2. Patch 2 (official RAE) — re-score offline, confirm headline numbers hold.
3. Patch 3 (negative control) — one cheap re-run of two arms.
4. Re-run `further-improvements/bootstrap_lift.py` at the molecule level for tight CIs.
