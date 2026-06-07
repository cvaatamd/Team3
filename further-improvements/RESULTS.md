# Further improvements — results

Post-hoc improvements to the 5-endpoint study in [`../results-fullexp/EXPERIMENT.md`](../results-fullexp/EXPERIMENT.md).
Everything here is isolated from the original run (patched source copy under `src/`, separate
configs/outputs). Heavy compute ran on GPU compute nodes via `sbatch`; the login node only submits.

## What changed

1. **Persisted per-molecule predictions** (`src/agents/ml_agent.py`): every run writes native-scale
   `(y_true, y_pred)` so any metric is recomputable offline and significance-testable at the
   molecule level — no retraining.
2. **Official RAE wired + verified**: `scripts/rescore_official.py` recomputes every cell from the
   stored predictions. Result: `max |official − range_normalized| = 0.00e+00` → the EXPERIMENT.md
   RAE numbers **are** the challenge-scoring RAE.
3. **Real transfer mechanisms**: in the original run `pretrain_finetune` / `frozen_embed` silently
   degraded to a from-scratch model (no checkpoint was ever produced). The patched agent now truly
   pretrains the Chemprop encoder on the auxiliary tasks (same multitask head shape as fine-tune, so
   the checkpoint reloads cleanly) and then transfers.

## Headline findings

**`pretrain_finetune` beats `mt_cotrain` in the few-shot regime** (paired bootstrap over 5 seeds,
Δ = old − new, positive ⇒ lower error). Significant (95% CI excludes 0) in 3/12 cells, never
significantly worse:

| endpoint | arm | n | mt_cotrain | pretrain_finetune | Δ (95% CI) | sig |
|---|---|---:|---:|---:|---|:--:|
| KSOL | both | 50 | 0.402 | 0.379 | +0.023 (+0.007, +0.038) | YES |
| KSOL | external | 100 | 0.402 | 0.381 | +0.021 (+0.002, +0.041) | YES |
| MPPB | both | 50 | 0.152 | 0.134 | +0.018 (+0.003, +0.033) | YES |

**Resolves the n=25 "collapse"** flagged in EXPERIMENT.md (old transfer arms were a degenerate,
aux-ignoring fallback → identical, high-variance, often worse than baseline). `pretrain_finetune`
uses the aux signal through the encoder and flips transfer to net-beneficial:

| endpoint (n=25) | baseline | old (collapsed) | pf +both | pf +external |
|---|---:|---:|---:|---:|
| MBPB | 0.165 | 0.162 | 0.116 | 0.139 |
| MPPB | 0.181 | 0.205 | 0.144 | 0.170 |
| KSOL | 0.414 | 0.429 | 0.361 | 0.372 |

**`frozen_embed` ruled out**: ridge on frozen embeddings is worse in all 12 cells and unstable on
KSOL (RAE 1.4–9.9). For these endpoints the encoder must adapt to the target assay.

## Files

- Patched code: `src/` (diff vs original: `ml_agent.py` pretrain/persist, `metrics.py` official RAE,
  `runner.py`/`contracts.py`/`planner.py`/`data_agent.py`/`harmonize.py` plumbing).
- Configs: `conf/fewshot-{pretrain_finetune,frozen_embed}.yaml`, `conf/smoke-*.yaml`.
- Launchers: `scripts/submit_regen.sh` + `scripts/regen.worker.sh` (run with `ADMET_USE_LLM=0`).
- Analysis: `scripts/rescore_official.py`, `scripts/compare_fewshot.py`, `scripts/bootstrap_lift_molecule.py`.
- Raw outputs: `outputs/compare_pretrain_finetune.csv`, `outputs/compare_frozen_embed.csv`,
  `outputs/rescore_pf.csv`; per-molecule preds in `results-fewshot/{pf,fe}/preds/`.

## Reproduce

```bash
# GPU sweep (compute nodes only; submits and exits):
ADMET_USE_LLM=0 bash further-improvements/scripts/submit_regen.sh \
  further-improvements/conf/fewshot-pretrain_finetune.yaml \
  $PWD/further-improvements/results-fewshot/pf

# offline analysis (light; inside the project container):
PYTHONPATH=further-improvements/src python further-improvements/scripts/rescore_official.py \
  --preds-dir further-improvements/results-fewshot/pf/preds
PYTHONPATH=further-improvements/src python further-improvements/scripts/compare_fewshot.py \
  --new-preds further-improvements/results-fewshot/pf/preds --label pretrain_finetune
```
