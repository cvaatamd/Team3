# Chemprop stability with external data — diagnosis and fixes

This document records the engineering investigation that took the main 240-job sweep from
**219/240 valid** (21 `external`-arm cells returning `rae=null`) to **240/240 valid**. It applies
to any run that uses Chemprop v2 multitask co-training on harmonized external sources under SLURM
array concurrency.

Companion docs: [RESULTS.md](RESULTS.md) (findings), [Plan.md](Plan.md) §2.4–§3 (design),
[Readme.md](Readme.md) (how to run).

---

## Symptom

During the first full sweep (`results/`):

| Observation | Detail |
|---|---|
| Failed cells | **21 / 240** — almost all **`external` arm**, mostly **MBPB** |
| Run JSON | `rae: null`, `n_test: 0`, `extra_metrics` all null |
| Logs | Training often reached `max_epochs=80` without a Python traceback |
| Confusing clue | The **`both` arm** used the **same external source** (`biogen/adme-fang-v1` human PPB
  proxy for mouse MBPB) and was **stable** (RAE ≈ 0.11) |

`n_test=0` means every `(y_true, y_pred)` pair was dropped — i.e. **all predictions were NaN**
(`src/eval/metrics.py` masks invalid pairs). The test labels themselves were fine.

---

## What was *not* the root cause (investigated and ruled out)

The debugging session tried several hypotheses before finding the real bug:

| Hypothesis | Why it seemed plausible | Why it failed |
|---|---|---|
| **Checkpoint race** | `FileNotFoundError` on shared `checkpoints/epoch=…-vN.ckpt` under concurrent array tasks | Real bug, but fixing it alone left MBPB `external` at `nan` |
| **Misaligned external source** | MBPB source is `species-transfer`, `n_overlap=0`, human→mouse PPB proxy | **`both` arm trains fine with the same source**; skipping the source made `external` target-only and still `nan` |
| **Learning rate too high** | `max_lr=1e-3` can explode gradients on sparse pools | Lowered to `2e-4` — **still `nan`** |
| **Degenerate aux values / logit ceiling** | Human PPB → aux head values pile up near the logit clip rail (~±6.9) | Insightful for MBPB aux quality, but not the mechanism that killed training |
| **Empty test set after harmonization** | Initial misread of `n_test=0` | Test split always has valid MBPB labels; predictions were NaN, not missing test rows |

---

## Three real issues (and fixes)

### 1. Empty-batch NaN in masked multitask loss — **primary fix**

**Root cause.** For sparse arms (`baseline`, `external`), `DataAgent.build_pool()` kept the full
ExpansionRx train frame (~5,300 rows) plus a few hundred harmonized external rows. After merging,
only a **small fraction** of rows had a label in **any** task column — e.g. for MBPB `external` at
n=250: ~250 target labels + ~194 aux labels among ~5,500 rows (**~95% all-NaN across tasks**).

Chemprop's masked multi-task loss (`batch_size=64`) then frequently sampled batches with **zero
valid labels** for every task → masked loss = **0/0 = NaN** → weights corrupted → all-NaN
predictions.

**Why `both` / `intra_task` were stable:** intra-task auxiliary heads (other ExpansionRx endpoints)
label nearly every row in the pool, so batches almost always contain valid targets.

**Fix** — drop fully-unlabeled rows before training (`src/agents/data_agent.py`, end of
`build_pool()`):

```python
labeled = df[task_cols].notna().any(axis=1)
if (~labeled).sum():
    df = df.loc[labeled].copy()
```

**Validation:** re-ran the 21 failed GPU cells; all recovered real RAE values. Pools shrank to
hundreds of labeled rows; jobs also ran faster.

**Provenance:** `TrainingPool.provenance["n_unlabeled_dropped"]` logs how many rows were removed.

---

### 2. Shared-checkpoint race under SLURM array concurrency

**Root cause.** Lightning's default `ModelCheckpoint` auto-versions filenames in a shared
`checkpoints/` directory. Concurrent array tasks deleted/overwrote each other's `.ckpt` files →
`FileNotFoundError` mid-`trainer.fit()`.

**Fix** (`src/models/chemprop_mt.py`):

- `enable_checkpointing=False` on the Lightning `Trainer` (we predict in-process right after `fit`;
  optional explicit save via `checkpoint_out` when needed).
- Pin `LightningEnvironment()` so interactive Cray/PMI allocations don't trigger
  `MPIEnvironment` / `PMI_Init returned -1` on LUMI.

---

### 3. Wrong FFN `output_transform` type

**Root cause.** `train_ds.normalize_targets()` returns an sklearn `StandardScaler`, but the Chemprop
FFN expects a chemprop `nn.Module` (`UnscaleTransform`). Passing the raw scaler caused incorrect
unscaling and could contribute to NaN predictions.

**Fix** (`src/models/chemprop_mt.py`):

```python
scaler = train_ds.normalize_targets()
val_ds.normalize_targets(scaler)
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
# … pass output_transform= to RegressionFFN
```

---

## Extra stability margin (kept, not sufficient alone)

These changes remain in the codebase as headroom for sparse multitask pools:

| Setting | Before | After | File |
|---|---|---|---|
| `max_lr` | `1e-3` | `2e-4` | `src/models/chemprop_mt.py` (`ChempropConfig`) |
| `grad_clip` | — | `1.0` | same |

Lowering LR alone did **not** fix the MBPB `external` NaN; the labeled-row filter did.

---

## External-source context (MBPB example)

For completeness — the harmonization log for the failing cells looked like:

```
source=biogen/adme-fang-v1  match_quality=species-transfer
calibration: n_overlap=0, slope=None, intercept=None, r2=None, aligned=False
merge_strategy=aux_head
```

The human plasma-protein-binding proxy is a weak cross-species match for mouse brain binding.
That affects **scientific interpretation** (muted external lift), but it was **not** why Chemprop
returned NaN once the empty-batch bug was fixed — the same source trains stably in the `both` arm.

---

## How to diagnose similar failures in future runs

1. **Check run JSON:** `rae is null` and `n_test == 0` → all-NaN predictions.
2. **Read worker log:** distinguish `FileNotFoundError` on `checkpoints/` (issue #2) from clean
   training completion with bad metrics (issue #1).
3. **Inspect pool shape** in run provenance / harmonization log:
   - `n_rows_total` vs sum of `n_per_task` — a huge gap means many all-NaN rows.
   - `n_unlabeled_dropped` should be > 0 for sparse arms after the fix.
4. **Compare arms:** if `both` is stable but `external` is NaN with the same external source, suspect
   empty-batch masked loss (issue #1), not source quality alone.

---

## Related fixes (same debugging session, not Chemprop training)

These were fixed while getting the sweep to 240/240 but are separate from Chemprop instability:

| Issue | Fix |
|---|---|
| Aitta narrative returned empty / non-JSON | `max_tokens` 1024 → 8192 in `conf/llm.yaml`; graceful fallback to deterministic report in `src/agents/llm.py` / `llm_overrides.py` |
| `collect --llm` crashed the whole pipeline | `LLMPlanner._write_report` catches narrative failures and writes the templated report instead |

---

## Files changed (summary)

| File | Change |
|---|---|
| `src/agents/data_agent.py` | Drop rows with no label in any task column (**primary NaN fix**) |
| `src/models/chemprop_mt.py` | `enable_checkpointing=False`, `UnscaleTransform`, `LightningEnvironment`, lower `max_lr`, `grad_clip` |
| `src/agents/llm.py`, `src/agents/llm_overrides.py` | LLM narrative fallback |
| `conf/llm.yaml` | Larger `max_tokens` for reasoning model JSON output |

---

## Timeline (Jun 5–6, 2026)

1. First sweep completes **219/240** — checkpoint races + NaN external cells mixed together.
2. Fix checkpoint race + re-run 21 cells → training clean, **still NaN** on MBPB external.
3. Try skip misaligned external source → **`both` disproves** source-as-cause; revert skip.
4. Try lower learning rate → **still NaN**.
5. Identify empty-batch masked loss → labeled-row filter → re-run → **240/240 valid**.

Full agent conversation: [RECOVERED_CHAT_HISTORY.md](RECOVERED_CHAT_HISTORY.md) (messages ~8–14).
