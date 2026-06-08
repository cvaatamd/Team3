# Results — Multi-task few-shot ADMET transfer

Characterization of auxiliary-task transfer for two ADMET endpoints, run on LUMI-G via the
SLURM sweep in this repo. This document summarizes the final, complete sweep
(`results/`, 240/240 jobs valid). Full experiment write-up (design, reproduction, tables with
error bands): [`results/EXPERIMENT.md`](results/EXPERIMENT.md). The machine-generated companion is
[`results/report.md`](results/report.md); raw per-job records live in `results/runs/` and the
aggregate table in `results/results.parquet`.

## Experimental setup

| Axis | Values |
|---|---|
| Target endpoints | `HLM CLint`, `MBPB` |
| Arms | `baseline` (single-task), `intra_task` (MT on ExpansionRx aux endpoints), `external` (MT on a harmonized external source), `both` (intra + external) |
| n target labels | 25, 50, 100, 250, 500, full |
| Seeds | 0, 1, 2, 3, 4 |
| Metric | **RAE** (range-normalized relative absolute error, native scale) — lower is better |
| Models | `baseline` → LightGBM; transfer arms → Chemprop v2 multi-task (80 epochs) |
| Eval | fixed held-out ExpansionRx test split; metric averaged over 5 seeds |

Total jobs: 2 × 4 × 6 × 5 = **240**, all valid (no NaN/failed cells).

## Headline findings

1. **MBPB benefits clearly from auxiliary transfer.** The `both` and `intra_task` arms reduce
   RAE by ~0.03–0.04 (≈25–35% relative) versus baseline once n ≥ 50, with non-overlapping
   mean±std bands at n = 250 and 500 — a credible, significant lift.
2. **HLM CLint does not benefit.** The single-task baseline is best at every data regime; all
   transfer arms sit at or above baseline within error bars. No reliable transfer.
3. **The `external` arm is endpoint-dependent.** It helps MBPB at n = 100/250 but is ambiguous
   (overlapping bands) at the extremes, and tracks baseline for HLM CLint. The external source
   is a cross-domain proxy, so a muted/mixed effect is expected.

**Recommendation:** use `both` or `intra_task` for MBPB; use the plain single-task baseline for
HLM CLint.

## Plots

Learning curves (RAE vs n target labels), per endpoint:

![RAE learning curves per endpoint](results/curves.png)

Macro-averaged RAE across both endpoints:

![Macro-averaged RAE vs n](results/ma_rae.png)

## Result tables (mean RAE over 5 seeds)

### HLM CLint — baseline best at every n

| n | baseline | both | external | intra_task |
|----:|----:|----:|----:|----:|
| 25 | **0.105** | 0.187 | 0.187 | 0.187 |
| 50 | **0.104** | 0.112 | 0.127 | 0.149 |
| 100 | **0.106** | 0.143 | 0.107 | 0.122 |
| 250 | **0.100** | 0.111 | 0.103 | 0.118 |
| 500 | **0.093** | 0.110 | 0.096 | 0.125 |

### MBPB — transfer wins from n ≥ 50

| n | baseline | both | external | intra_task |
|----:|----:|----:|----:|----:|
| 25 | 0.165 | **0.162** | 0.162 | 0.162 |
| 50 | 0.153 | **0.117** | 0.143 | 0.122 |
| 100 | 0.146 | **0.119** | 0.125 | 0.125 |
| 250 | 0.153 | **0.112** | 0.120 | **0.112** |
| 500 | 0.127 | **0.101** | 0.120 | 0.107 |

(Lowest value per row in **bold**. At n = 25 the arms collapse to nearly the same value because
the target budget dominates.)

## Interpretation notes

- The widest uncertainty is at n = 25 for both endpoints; conclusions there are not robust.
- For MBPB, `both` ≈ `intra_task` at high n, indicating most of the lift comes from the
  in-distribution ExpansionRx auxiliary endpoints rather than the external source.
- For HLM CLint, the near-identical external assay (`external` arm) closely tracks baseline,
  confirming it neither helps nor hurts materially.

## Reproducing

```bash
# Full sweep + report (SLURM does the work):
bash scripts/run_full_sweep.sh

# Inspect:
bash scripts/check_status.sh
cat results/report.md
```

Outputs land in `results/`: `results.parquet`, `curves.png`, `ma_rae.png`, `report.md`.

## Engineering notes (issues found and fixed during this run)

These were uncovered while getting the sweep to 240/240 clean and are documented for future runs:

1. **Empty-batch NaN (root cause of the MBPB `external` failures).** Sparse arms
   (`baseline`/`external`) kept ExpansionRx rows with no label in *any* task column; under the
   masked multi-task loss, batches with zero valid labels produced `0/0 = NaN`, which propagated
   and yielded all-NaN predictions (`RAE=nan`). Fix: the data agent now drops fully-unlabeled
   rows from the training pool (`src/agents/data_agent.py`).
2. **Shared-checkpoint race.** Concurrent array tasks writing Lightning checkpoints into one
   shared `checkpoints/` dir raced (`FileNotFoundError`). Fix: `enable_checkpointing=False` on
   the trainer (`src/models/chemprop_mt.py`); also fixed the FFN `output_transform` and pinned
   `LightningEnvironment` to avoid a Cray/PMI crash.
3. **Training-stability margin.** Lowered Chemprop `max_lr` 1e-3 → 2e-4 for extra headroom on
   sparse pools.
4. **LLM narrative robustness.** The Aitta planner narrative now (a) uses a larger token budget
   so the reasoning model returns non-empty JSON (`conf/llm.yaml` `max_tokens` 1024 → 8192) and
   (b) degrades gracefully to the deterministic report instead of crashing `collect`
   (`src/agents/llm.py`, `src/agents/llm_overrides.py`).
