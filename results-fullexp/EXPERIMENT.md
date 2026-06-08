# Full experiment — 5-endpoint ADMET transfer study (Aitta agent on LUMI)

Run date: 2026-06-06 · LUMI-G · `project_462001520`.
Self-contained, separate from earlier runs (`results/`, `results-rounds/`); nothing here depends
on previous results. Companion docs: [OVERVIEW.md](../OVERVIEW.md), [Readme.md](../Readme.md).

## Project context

**Thesis.** For a sparse target ADMET endpoint, auxiliary training data from (a) richer ExpansionRx
endpoints (cross-task) and (b) external ADMET datasets (cross-dataset) lowers prediction error most
in the low-data regime, and the size of that lift depends on how well the auxiliary assay matches the
target. We demonstrate this with subsampled learning curves and characterize the pattern.

**System.** An agentic pipeline (Planner → Data agent → ML agent) over a deterministic toolset. The
data agent curates and **harmonizes** target + auxiliary sources (unit reconciliation, calibration
checks, co-train-vs-pool merge decisions); the ML agent picks a transfer mechanism and trains
(LightGBM single-task baseline vs. Chemprop v2 multitask); the planner owns the sweep and writes the
interpretation. Evaluation matches the OpenADMET–ExpansionRx challenge: **RAE / MA-RAE** on the
official time-split, with the held-out test touched once at scoring.

**This run** instantiates that system across five endpoints spanning the data-richness spectrum and
three external match types, to map *where, and from what source, transfer actually helps*. Full
build spec and rationale: [../Plan.md](../Plan.md).

## Objective

Characterize **where, and from what source, auxiliary-data transfer helps few-shot ADMET
prediction**, across **five endpoints spanning the data-richness spectrum** and three external
match types. Each endpoint is a full sweep (4 arms × 6 data sizes × 5 seeds), run by the standalone
Aitta agent loop (Planner → Data agent → ML agent → Planner report).

## Design

5 rounds (one endpoint each), all run as **parallel SLURM arrays** with a dependent collect job:

| Round | Endpoint | Labels (train) | External source | Match quality |
|---|---|---:|---|---|
| hlm | `HLM CLint` | 3759 | Biogen `LOG_HLM_CLint` | near-identical |
| mbpb | `MBPB` | 975 | Biogen `LOG_HPPB` | species-transfer (weak) |
| ksol | `KSOL` | 5128 | Biogen `LOG_SOLUBILITY` | protocol-caveat (µg/mL→µM) |
| mppb | `MPPB` | 1302 | Biogen `LOG_HPPB` | species-transfer |
| caco2eff | `Caco-2 Permeability Efflux` | 2161 | Biogen `LOG_MDR1-MDCK_ER` | analogous (diff. cell line) |

Arms: `baseline` (LightGBM single-task), `+ intra_task` (Chemprop multitask on ExpansionRx aux
endpoints), `+ external` (multitask on the harmonized external source), `+ both`.

| Axis | Value |
|---|---|
| n target labels | 25, 50, 100, 250, 500, **full** |
| Seeds | 0,1,2,3,4 → mean RAE ± std |
| Jobs / round | 4 × 6 × 5 = **120** (600 total) |
| Metric | RAE (range-normalized, native scale; **lower is better**) |
| Models | `baseline`→LightGBM; transfer arms→Chemprop v2 multitask |
| Eval | fixed held-out ExpansionRx test split, scored once |
| LLM | Aitta `openai/gpt-oss-120b`; decisions cached at `results/llm_cache` |

## Execution / SLURM

Launched with `scripts/run_fullexp.sh` using `conf/slurm-fullexp.yaml`:
**array_concurrency: 100**, `max_array_tasks: 36` → each round = 36 array tasks striding 4 jobs;
5 rounds × 36 + 5 collect = **185 submitted jobs** (under LUMI small-g's ~200 limit), with up to
~180 GPU jobs running concurrently across the five arrays.

Job IDs are recorded in [`JOBIDS.txt`](JOBIDS.txt). At submit time:

| Round | Sweep | Collect (afterany) | Output |
|---|---|---|---|
| hlm | 19073763 | 19073764 | `results-fullexp/hlm/` |
| mbpb | 19073778 | 19073779 | `results-fullexp/mbpb/` |
| ksol | 19073780 | 19073781 | `results-fullexp/ksol/` |
| mppb | 19073782 | 19073783 | `results-fullexp/mppb/` |
| caco2eff | 19073786 | 19073787 | `results-fullexp/caco2eff/` |

**Disconnection-safe:** the launcher only submits, then exits; SLURM runs the sweeps and the
chained collect jobs independently of any login session. You can log out immediately after
submitting and the reports + plots will still be produced.

---

## Step-by-step reproduction (copy-paste)

> **Hands-on runbook:** [`RUNBOOK.md`](./RUNBOOK.md) — launch, monitor, single-round submit,
> re-run/resume, alternative metrics, troubleshooting.

Run from the project root on a **LUMI login node**. Everything is standalone.

```bash
# 0) One-time: get a 24h Aitta token from https://aitta-auth.csc.fi/myToken
echo 'PASTE_YOUR_TOKEN_HERE' > conf/.aitta_token && chmod 600 conf/.aitta_token

# 1) Preflight: confirm Aitta is reachable and the plan enumerates (no jobs launched)
bash scripts/run_agent.sh --check

# 2) (optional) Preview what will be submitted — renders sbatch scripts, submits nothing
bash scripts/run_fullexp.sh --dry-run

# 3) Launch the full 5-round experiment (disconnection-safe; logs out fine after this)
nohup bash scripts/run_fullexp.sh &>results-fullexp/submit.log &
#    ...or simply:  bash scripts/run_fullexp.sh

# 4) Monitor (any time, from any session)
squeue -u $USER                 # R = running, blank = done
bash scripts/check_status.sh    # higher-level progress
cat results-fullexp/JOBIDS.txt  # the sweep/collect job ids per round

# 5) Read each round's result when its collect job finishes
cat results-fullexp/hlm/report.md
cat results-fullexp/mbpb/report.md
cat results-fullexp/ksol/report.md
cat results-fullexp/mppb/report.md
cat results-fullexp/caco2eff/report.md

# 6) (only if needed) re-aggregate one round's report without re-training
#    (e.g. after a tweak); standalone, no GPU:
RESULTS_DIR=results-fullexp/ksol bash scripts/submit_collect.sh
```

Config files for this experiment (edit to change scope):
`conf/fullexp-{hlm,mbpb,ksol,mppb,caco2eff}.yaml`, `conf/slurm-fullexp.yaml`,
launcher `scripts/run_fullexp.sh`.

Per-round outputs: `results-fullexp/<round>/{results.parquet, curves.png, ma_rae.png, report.md, runs/}`.

---

## Results

All 600 jobs completed (no failures). Tables show **mean RAE ± std over 5 seeds** (lower is
better); **bold** = best arm at that data size. Arms: `+intra` = ExpansionRx multitask,
`+external` = Biogen multitask, `+both` = both auxiliary sources.

### hlm — HLM CLint (rich, n=3759)
![HLM CLint learning curves](hlm/curves.png)

| n | baseline | +intra | +external | +both |
|---|---|---|---|---|
| 25 | **0.105±0.004** | 0.187±0.165 | 0.187±0.165 | 0.187±0.165 |
| 50 | **0.104±0.005** | 0.110±0.017 | 0.132±0.042 | 0.111±0.015 |
| 100 | **0.106±0.005** | 0.145±0.038 | 0.107±0.006 | 0.119±0.030 |
| 250 | **0.100±0.005** | 0.121±0.020 | 0.107±0.005 | 0.115±0.011 |
| 500 | **0.093±0.005** | 0.120±0.013 | 0.098±0.004 | 0.162±0.063 |
| full | **0.085±0.002** | 0.134±0.052 | 0.093±0.007 | 0.122±0.019 |

Single-task LightGBM wins at **every** size — HLM CLint already has enough labels that transfer
only adds noise. Negative result (transfer is the wrong move here).

### mbpb — MBPB (sparse, n=975)
![MBPB learning curves](mbpb/curves.png)

| n | baseline | +intra | +external | +both |
|---|---|---|---|---|
| 25 | 0.165±0.010 | **0.162±0.020** | 0.162±0.020 | 0.162±0.020 |
| 50 | 0.157±0.014 | 0.127±0.010 | 0.152±0.029 | **0.122±0.016** |
| 100 | 0.146±0.013 | 0.129±0.019 | 0.133±0.021 | **0.127±0.022** |
| 250 | 0.153±0.006 | **0.121±0.013** | 0.130±0.020 | 0.121±0.020 |
| 500 | 0.127±0.006 | 0.111±0.009 | 0.109±0.023 | **0.102±0.012** |
| full | 0.121±0.011 | 0.100±0.009 | **0.096±0.008** | 0.103±0.016 |

Transfer helps strongly and consistently (≈20–22%), including at full data. Best low-data arm
combines both sources; external alone is best at full.

### ksol — KSOL (richest, n=5128)
![KSOL learning curves](ksol/curves.png)

| n | baseline | +intra | +external | +both |
|---|---|---|---|---|
| 25 | **0.414±0.036** | 0.429±0.075 | 0.429±0.075 | 0.429±0.075 |
| 50 | 0.424±0.020 | 0.422±0.088 | **0.400±0.029** | 0.402±0.047 |
| 100 | 0.427±0.018 | 0.543±0.369 | 0.402±0.039 | **0.387±0.042** |
| 250 | 0.396±0.025 | 0.355±0.024 | **0.331±0.020** | 0.374±0.037 |
| 500 | 0.372±0.016 | 0.345±0.013 | **0.341±0.024** | 0.343±0.021 |
| full | 0.329±0.008 | 0.310±0.014 | **0.270±0.013** | 0.287±0.009 |

High absolute RAE (KSOL is intrinsically hard), but the harmonized Biogen solubility source is the
best arm from n=250 upward and cuts RAE **18%** at full data — the unit conversion (µg/mL→µM) paid off.

### mppb — MPPB (mid, n=1302)
![MPPB learning curves](mppb/curves.png)

| n | baseline | +intra | +external | +both |
|---|---|---|---|---|
| 25 | **0.181±0.018** | 0.205±0.020 | 0.205±0.020 | 0.205±0.020 |
| 50 | 0.170±0.012 | 0.156±0.027 | 0.179±0.024 | **0.152±0.020** |
| 100 | 0.174±0.010 | 0.144±0.013 | 0.152±0.009 | **0.144±0.024** |
| 250 | 0.167±0.011 | 0.147±0.020 | 0.148±0.011 | **0.137±0.021** |
| 500 | 0.156±0.008 | 0.136±0.018 | 0.136±0.012 | **0.123±0.008** |
| full | 0.161±0.002 | 0.127±0.015 | **0.118±0.012** | 0.129±0.009 |

Largest transfer payoff of the study: external human-PPB cuts RAE **27%** at full data; `+both`
leads through the few-shot/mid regime.

### caco2eff — Caco-2 Permeability Efflux (mid, n=2161)
![Caco-2 Efflux learning curves](caco2eff/curves.png)

| n | baseline | +intra | +external | +both |
|---|---|---|---|---|
| 25 | **0.054±0.054** | 0.153±0.142 | 0.153±0.142 | 0.153±0.142 |
| 50 | 0.030±0.001 | 0.029±0.001 | 0.029±0.001 | **0.029±0.001** |
| 100 | 0.030±0.000 | 0.029±0.000 | 0.029±0.001 | **0.029±0.001** |
| 250 | 0.030±0.001 | 0.029±0.001 | 0.030±0.000 | **0.029±0.001** |
| 500 | 0.029±0.001 | 0.029±0.001 | 0.030±0.001 | **0.029±0.001** |
| full | 0.028±0.000 | 0.029±0.001 | 0.029±0.001 | **0.028±0.000** |

Effectively a ceiling case — every arm reaches RAE ≈0.03 by n=50, so there is no headroom for
transfer to exploit (the n=25 baseline's 0.054±0.054 is a single-seed fluke, note the variance).

### Cross-round synthesis

Best transfer arm vs. baseline (negative % = transfer worse; positive = transfer better):

| Endpoint | Labels | few-shot (n=50) | full data |
|---|---:|---|---|
| HLM CLint | 3759 | +intra **−6.4%** (baseline wins) | +external **−9.2%** (baseline wins) |
| Caco-2 Efflux | 2161 | +both +2.1% | +both +1.0% |
| MPPB | 1302 | +both **+10.7%** | +external **+26.9%** |
| MBPB | 975 | +both **+22.4%** | +external **+20.9%** |
| KSOL | 5128 | +external +5.6% | +external **+18.0%** |

**Takeaways**

1. **Transfer value tracks baseline headroom, not raw data size.** Where single-task LightGBM is
   already strong, transfer is neutral-to-harmful: HLM CLint (baseline best at every n) and the
   near-saturated Caco-2 Efflux (everything ≈0.03). Where the baseline is weak — MBPB, MPPB, and the
   intrinsically hard KSOL — multitask transfer delivers large, repeatable gains (≈18–27%).
2. **External assays carry real signal even with imperfect match.** At full data the **external
   Biogen source is the single best arm in 3/5 rounds** (MBPB, KSOL, MPPB) and second on HLM. This
   holds despite only species-transfer (MBPB/MPPB) or protocol-caveat (KSOL) matches, validating the
   agent's harmonization (incl. the KSOL µg/mL→µM conversion).
3. **In the few-shot regime, combine sources.** `+both` is the best arm at n=50 for MBPB, MPPB and
   Caco-2; intra-task ExpansionRx co-training is the safer early signal before external dominates at
   larger n.
4. **Caveats.** At n=25 the three transfer arms collapse to identical, high-variance values (too few
   target labels to differentiate) — treat n=25 as unreliable. Per-round LLM narratives are in each
   `results-fullexp/<round>/report.md`.

**Decision rule the agent learned:** spend effort on transfer when the single-task baseline is
weak (sparse or intrinsically hard endpoints); prefer the harmonized external assay at moderate-to-
full data and add intra-task co-training in the few-shot regime; skip transfer when the baseline is
already strong or the task is near-saturated.

## Alternative metrics (not just RAE)

Every run stores **`rae`, `mae`, `rmse`, `r2`, `spearman`** in `results.parquet`, so other views
need **no re-training**. Regenerate plots + report for any metric (writes
`curves_<metric>.png`, `ma_<metric>.png`, `report_<metric>.md`, leaving the RAE outputs intact):

```bash
# no LLM, no token, no GPU training (just re-reads runs/):
RESULTS_DIR=results-fullexp/mppb METRIC=r2 bash scripts/submit_collect.sh   # SLURM
# or inside the container directly:
python -m experiment.cli collect --results-dir results-fullexp/mppb --metric r2   # rae|mae|rmse|r2|spearman
```

Full-data (n=full) means over 5 seeds, pulled from the existing parquet:

**R² (higher is better)**

| endpoint | baseline | intra | external | both |
|---|---|---|---|---|
| HLM CLint | 0.153 | -2.611 | 0.047 | -1.456 |
| MBPB | 0.109 | 0.371 | **0.429** | 0.339 |
| KSOL | -0.425 | -0.470 | **-0.033** | -0.170 |
| MPPB | -0.120 | 0.258 | **0.378** | 0.217 |
| Caco-2 Efflux | -0.098 | -0.120 | -0.118 | -0.078 |

**Spearman ρ (higher is better)**

| endpoint | baseline | intra | external | both |
|---|---|---|---|---|
| HLM CLint | **0.561** | 0.470 | 0.483 | 0.480 |
| MBPB | 0.749 | 0.823 | 0.822 | **0.834** |
| KSOL | 0.485 | **0.544** | 0.515 | 0.533 |
| MPPB | 0.487 | **0.738** | 0.723 | 0.718 |
| Caco-2 Efflux | **0.626** | 0.524 | 0.538 | 0.592 |

**RMSE, native scale (lower is better)**

| endpoint | baseline | intra | external | both |
|---|---|---|---|---|
| HLM CLint | **39.87** | 74.17 | 42.26 | 67.49 |
| MBPB | 6.20 | 5.21 | **4.97** | 5.32 |
| KSOL | 136.84 | 138.55 | **116.44** | 123.92 |
| MPPB | 14.96 | 12.09 | **11.11** | 12.49 |
| Caco-2 Efflux | 42.64 | 43.06 | 43.02 | **42.24** |

A different *RAE definition* is also pluggable (`range_normalized` (default), `vs_mean`, or the
`official` stub) via `rae_definition` in `src/eval/metrics.py`; changing it recomputes from the
stored predictions.

### Winner matrix (best arm)

Best arm per endpoint × metric. Error metrics (RAE/MAE/RMSE) and R² move together; **Spearman
(ranking) is the outlier** — it favours transfer more broadly, especially few-shot.

**Full data**

| Endpoint | RAE | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|---|
| HLM CLint | baseline | baseline | baseline | baseline | baseline |
| MBPB | external | external | external | external | both |
| KSOL | external | external | external | external | intra |
| MPPB | external | external | external | external | intra |
| Caco-2 Efflux | both | both | both | both | baseline |

**Few-shot (n=50)**

| Endpoint | RAE | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|---|
| HLM CLint | baseline | baseline | baseline | baseline | **both** |
| MBPB | both | both | both | both | both |
| KSOL | external | external | external | external | intra |
| MPPB | both | both | both | both | both |
| Caco-2 Efflux | both | both | both | both | **both** |

### Cross-metric conclusions

1. **The magnitude metrics agree.** RAE, MAE, RMSE and R² pick the *same* winner in 9/10 endpoint×regime
   cells: external assays minimise error at full data on MBPB/KSOL/MPPB, `+both` leads few-shot, and
   single-task baseline wins on HLM CLint. The headline RAE conclusions are **not an artifact of the
   metric** — they reproduce on absolute-error (MAE/RMSE) and variance-explained (R²) scales.
2. **Ranking (Spearman) is more transfer-friendly, especially few-shot.** Transfer improves compound
   *ordering* even where it doesn't cut error: HLM CLint and Caco-2 Efflux — both "don't-transfer"
   endpoints on error — still get a Spearman lift from `+both` at n=50 (HLM 0.32→0.41, Caco-2
   0.34→0.52). For triage/ranking use-cases, transfer helps on **all five** endpoints in the
   few-shot regime.
3. **At full data the two objectives split.** External assays maximise calibrated accuracy
   (RAE/MAE/RMSE/R²), but **intra-task ExpansionRx co-training maximises rank correlation** on KSOL
   and MPPB (Spearman). Choose the auxiliary source by objective: external for magnitude, intra-task
   for ranking.
4. **HLM CLint stays "don't transfer"** on every error metric and R², and even on ranking at full
   data — only its few-shot ranking benefits marginally.
5. **Caco-2 Efflux is range-saturated, not error-saturated.** Tiny RAE (~0.03) but large native RMSE
   (~42) and negative R² — the low RAE is a dynamic-range artifact. Transfer barely moves its error
   but clearly improves few-shot ranking. Reading one metric (RAE) alone would have hidden both facts.

### All learning curves, by metric

Per-endpoint curves (arms as lines, ±1 std over seeds). RAE curves are in the per-round sections
above; the other four metrics follow.

**MAE (lower is better)**

| | |
|---|---|
| ![hlm MAE](hlm/curves_mae.png) | ![mbpb MAE](mbpb/curves_mae.png) |
| ![ksol MAE](ksol/curves_mae.png) | ![mppb MAE](mppb/curves_mae.png) |
| ![caco2eff MAE](caco2eff/curves_mae.png) | |

**RMSE (lower is better)**

| | |
|---|---|
| ![hlm RMSE](hlm/curves_rmse.png) | ![mbpb RMSE](mbpb/curves_rmse.png) |
| ![ksol RMSE](ksol/curves_rmse.png) | ![mppb RMSE](mppb/curves_rmse.png) |
| ![caco2eff RMSE](caco2eff/curves_rmse.png) | |

**R² (higher is better)**

| | |
|---|---|
| ![hlm R2](hlm/curves_r2.png) | ![mbpb R2](mbpb/curves_r2.png) |
| ![ksol R2](ksol/curves_r2.png) | ![mppb R2](mppb/curves_r2.png) |
| ![caco2eff R2](caco2eff/curves_r2.png) | |

**Spearman ρ (higher is better)**

| | |
|---|---|
| ![hlm Spearman](hlm/curves_spearman.png) | ![mbpb Spearman](mbpb/curves_spearman.png) |
| ![ksol Spearman](ksol/curves_spearman.png) | ![mppb Spearman](mppb/curves_spearman.png) |
| ![caco2eff Spearman](caco2eff/curves_spearman.png) | |

---

## Update — further improvements (post-hoc, statistically validated)

A second pass (fully isolated in [`../further-improvements/`](../further-improvements/) — **nothing in this
run was modified**) added reproducibility/statistics tooling and a *real* transfer mechanism, then
re-evaluated the few-shot regime with paired significance tests on a GPU sweep (LLM disabled, so the
mechanism is fixed rather than agent-chosen).

**What was added**

1. **Per-molecule predictions are now persisted.** Every run writes native-scale `(y_true, y_pred)`,
   so any metric can be recomputed and significance-tested offline — no retraining.
2. **Official RAE wired and verified.** Recomputing every cell offline from the stored predictions
   gives `max |official − range_normalized| = 0.00e+00` → the headline RAE numbers above **are**
   exactly the challenge-scoring RAE (now confirmed, not assumed). The `official` stub referenced in
   *Alternative metrics* is no longer a stub.
3. **Two transfer mechanisms the original sweep never truly ran were implemented for real.** In this
   run `pretrain_finetune` and `frozen_embed` silently degraded to a from-scratch model (nothing ever
   produced a pretrained checkpoint), which is why only `gbm_baseline` + `mt_cotrain` appear above.
   The patched ML agent now genuinely pretrains the Chemprop encoder on the auxiliary tasks and then
   transfers. Controlled sweep: 3 weak endpoints × {external, both} × n∈{25, 50, 100} × 5 seeds
   (270 GPU jobs, 0 failures).

### `pretrain_finetune` — a real few-shot win

Pretrain the encoder on the auxiliary heads, then fine-tune on the sparse target. Paired against the
original `mt_cotrain` per `(endpoint, arm, n, seed)`; **Δ = old − new** (positive ⇒ *lower* error);
95% CI from a 10k paired bootstrap over the 5 seeds.

| endpoint | arm | n | `mt_cotrain` RAE | `pretrain_finetune` RAE | Δ (95% CI) | seeds↑ | sig |
|---|---|---:|---:|---:|---|:--:|:--:|
| KSOL | both | 50 | 0.402 | **0.379** | +0.023 (+0.007, +0.038) | 4/5 | **YES** |
| KSOL | external | 100 | 0.402 | **0.381** | +0.021 (+0.002, +0.041) | 5/5 | **YES** |
| MPPB | both | 50 | 0.152 | **0.134** | +0.018 (+0.003, +0.033) | 4/5 | **YES** |
| MBPB | external | 50 | 0.152 | 0.142 | +0.010 (−0.006, +0.032) | 4/5 | ns |
| MBPB | external | 100 | 0.133 | 0.123 | +0.009 (−0.007, +0.031) | 2/5 | ns |
| MPPB | both | 100 | 0.144 | 0.131 | +0.013 (−0.007, +0.033) | 3/5 | ns |
| KSOL | external | 50 | 0.400 | 0.386 | +0.015 (−0.007, +0.036) | 3/5 | ns |
| MBPB | both | 50 | 0.122 | 0.122 | +0.000 (−0.010, +0.011) | 3/5 | ns |
| MBPB | both | 100 | 0.127 | 0.128 | −0.000 (−0.017, +0.017) | 2/5 | ns |
| MPPB | external | 50 | 0.179 | 0.178 | +0.001 (−0.008, +0.012) | 2/5 | ns |
| KSOL | both | 100 | 0.387 | 0.409 | −0.022 (−0.071, +0.020) | 2/5 | ns |
| MPPB | external | 100 | 0.152 | 0.158 | −0.006 (−0.011, +0.000) | 1/5 | ns |

`pretrain_finetune` **significantly lowers RAE in 3/12 cells** (CI excludes 0) and is **never
significantly worse in any cell**. The gains are modest in magnitude (~0.02 RAE) but real and paired.

### It resolves the n=25 "collapse" (Caveat 4 above)

The original flagged n=25 as unreliable: the three transfer arms collapsed to one identical,
high-variance value because the few-shot fallback **ignored the auxiliary data**. Because
`pretrain_finetune` injects the aux signal through the *encoder* before it ever sees the 25 target
labels, the arms differentiate and error drops sharply — flipping transfer from net-*harmful* to
net-*beneficial* vs the single-task baseline:

| endpoint (n=25) | baseline | old transfer (all arms collapsed) | `pretrain_finetune` +both | `pretrain_finetune` +external |
|---|---:|---:|---:|---:|
| MBPB | 0.165 | 0.162 | **0.116** | 0.139 |
| MPPB | 0.181 | 0.205 *(worse than baseline)* | **0.144** | 0.170 |
| KSOL | 0.414 | 0.429 *(worse than baseline)* | **0.361** | 0.372 |

(n=25 is reported descriptively, not bootstrapped: the original n=25 arms were a degenerate,
aux-ignoring fallback, so there is no matched `mt_cotrain` arm to pair against.)

### `frozen_embed` — ruled out

A ridge head on the *frozen* pretrained embeddings is worse in **all 12 cells** and unstable on KSOL
(RAE blows up to 1.4–9.9). Clean negative result: for these endpoints, fine-tuning the encoder beats
freezing it — the encoder needs to adapt to the target assay.

**Reproduce / inspect.** Patched source copy, configs, sbatch launchers, and analysis scripts
(`rescore_official.py`, `compare_fewshot.py`) live in [`../further-improvements/`](../further-improvements/);
raw tables in `further-improvements/outputs/` and per-molecule predictions in
`further-improvements/results-fewshot/{pf,fe}/preds/`. The heavy work ran on GPU compute nodes via
`sbatch`; nothing intensive runs on the login node.

---

# Appendix — Build plan / full spec

Below is the complete `Plan.md` build spec verbatim, embedded here so this document ships
standalone. (Original: [../Plan.md](../Plan.md).)

---

# Multi-task few-shot transfer learning for ADMET — build plan

A spec for an agentic system that demonstrates **when external and cross-task data lift few-shot
ADMET prediction**, evaluated on the OpenADMET–ExpansionRx challenge test split.

This document is the brief for Claude Code. Build the deterministic toolset first and get one
target endpoint working end-to-end, then wrap the three agents around it. Do **not** start with the
agent graph.

---

## 0. Thesis and what "done" looks like

**Thesis.** For a sparse target ADMET endpoint, training data from (a) richer ExpansionRx endpoints
(cross-task) and (b) external ADMET datasets (cross-dataset) lowers error most in the low-data
regime, and the size of that lift depends on how well the auxiliary assay matches the target. We
demonstrate this with subsampled learning curves and explain the pattern.

**Primary deliverables**
1. `n`-vs-RAE learning curves per endpoint, with error bands over seeds, for four arms (defined in §4).
2. An aggregate `n`-vs-MA-RAE curve (the challenge's headline metric).
3. A written characterization — produced by the planner agent — of *which* auxiliary tasks transfer,
   *in what data regime*, and *by how much*.
4. A reproducible agentic pipeline (planner / data / ML) that produced all of the above.

**Definition of done for the MVP.** Two target endpoints fully run through the sweep:
`HLM CLint` (strong external match) and one of `MBPB`/`Caco-2 Papp A>B` (weak/absent external match),
showing the contrast. Everything else is upside.

---

## 1. Scoring — match this exactly

- **Metric: MA-RAE** (macro-averaged relative absolute error). Per endpoint, the MAE is normalized by
  that endpoint's dynamic range on the test set, then averaged across endpoints. **Verify the exact
  RAE definition against the official scoring code** in the challenge HuggingFace Space
  (`openadmet/OpenADMET-ExpansionRx-Challenge`) before trusting any number — "relative absolute error"
  is ambiguous (range-normalized MAE vs. error relative to a mean predictor). Implement it as a
  pluggable metric so the definition can be swapped.
- **Split: official time-split.** Use the released splits directly; do not re-split.
  `train` = 5.33k molecules, `test` = 2.28k molecules, both now labeled (challenge closed 2026-01-19).
- Keep MAE / RMSE / R² / Spearman alongside RAE as diagnostics, but RAE/MA-RAE is the curve axis.

**Eval hygiene (non-negotiable).** The competition `test` split is touched exactly once, at evaluation.
Subsampling, model selection, and any hyperparameter tuning happen on the `train` split only
(carve an internal validation fold out of `train`). Never let test data influence a training pool.

---

## 2. Data

### 2.1 ExpansionRx (target dataset)

Source: `openadmet/openadmet-expansionrx-challenge-data` on HuggingFace (CC-BY-4.0).

```python
from datasets import load_dataset
train_df = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="train").to_pandas()
test_df  = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="test").to_pandas()
```

Use the `default` (ML-ready, in-range only) subset, not `raw` (which carries `>`/`<` censored
modifiers). The data is sparse: each molecule has labels for only a subset of endpoints.

**Columns:** `Molecule Name`, `SMILES`, then the 9 endpoint columns.

| Column | Endpoint | Units | Suggested transform |
|---|---|---|---|
| `LogD` | Lipophilicity | log unit | none (already log) |
| `KSOL` | Kinetic solubility | µM | log10 |
| `HLM CLint` | Human liver microsomal clearance | mL/min/kg | log10 |
| `MLM CLint` | Mouse liver microsomal clearance | mL/min/kg | log10 |
| `Caco-2 Permeability Papp A>B` | Passive permeability | 1e-6 cm/s | log10 |
| `Caco-2 Permeability Efflux` | Efflux ratio | ratio | log10 |
| `MPPB` | Mouse plasma protein binding | % unbound | logit(fu), fu=%/100 |
| `MBPB` | Mouse brain protein binding | % unbound | logit(fu) |
| `MGMB` | Mouse gastrocnemius muscle binding | % unbound | logit(fu) |

(The `raw` subset additionally has Rat Liver Microsomal `RLM CLint` (mL/min/kg), not part of the
official 9 — useful as an extra cross-task auxiliary.)

Train models in transformed space; **invert before computing RAE** so the metric is on the
challenge's native scale. Clip fraction-unbound to e.g. [0.001, 0.999] before logit. Handle zeros/
near-zeros in clearance/solubility with a small floor before log10.

**Coverage is the experiment.** Profile per-endpoint label counts on load. Expect `HLM/MLM CLint` and
`KSOL` to be the richest (natural *source* tasks) and `MBPB`/`MGMB`/`Caco-2` to be sparse (natural
*target* tasks). The data agent uses this profile to pick source/target pairings.

### 2.2 Known data-quality traps (from the challenge post-mortem)

Bake these into the data agent as flags:
- **Compound ID is a time proxy** (`E-00xxxxx` increasing ≈ later in the campaign). Use it to make
  subsampling respect temporal order (sample "earlier" compounds as history) and to reason about the
  time-split.
- **Solubility distribution shift in the first ~15% of compounds** (an assay-concentration artifact).
  Top teams improved by removing that slice. Make exclusion a toggle and test its effect.
- **HLM clearance has a large gap toward the end of the training set** → uneven availability across
  time-based splits. Be aware when subsampling HLM.

### 2.3 External sources (registry for the data agent)

The data agent draws from a **curated registry**, not the open web (provenance/licensing). Endpoint
mapping onto ExpansionRx targets, best first:

| ExpansionRx target | External source | Access | Match quality |
|---|---|---|---|
| `HLM CLint` (mL/min/kg) | Biogen/Fang `LOG HLM_CLint (mL/min/kg)` | Polaris `biogen/adme-fang-v1` | Near-identical (same assay, same units, already log) |
| `RLM CLint` (raw) | Biogen/Fang `LOG RLM_CLint (mL/min/kg)` | Polaris `biogen/adme-fang-v1` | Strong |
| `Caco-2 Efflux` | Biogen/Fang `LOG MDR1-MDCK ER` | Polaris `biogen/adme-fang-v1` | Analogous (different cell line) |
| `KSOL` (µM) | Biogen/Fang `LOG SOLUBILITY PH 6.8 (ug/mL)` | Polaris `biogen/adme-fang-v1` | Needs µg/mL→µM (MW) + protocol caveat |
| `MPPB` (mouse) | Biogen/Fang `hPPB` / `rPPB` (% unbound) | Polaris `biogen/adme-fang-v1` | Species transfer (human/rat→mouse) |
| `LogD` | ChEMBL / TDC lipophilicity | TDC / ChEMBL | Available, noisy/aggregated (stretch) |
| `Caco-2 Papp A>B` | TDC `Caco2_Wang` | `PyTDC` | Different system |
| `MBPB`, `MGMB` | — none public — | — | **Intra-dataset transfer only** |

Secondary source to wire in if time allows: **ASAP–Polaris–OpenADMET antiviral ADMET** (Polaris),
another sparse lead-op ADMET set.

Biogen/Fang: ~3,521 compounds, 6 endpoints (human/rat liver microsomal stability, MDR1-MDCK efflux
ratio, solubility, human/rat plasma protein binding). Already in log space.

```python
# verify current Polaris client API; login may be required for some datasets
import polaris as po
ds = po.load_dataset("biogen/adme-fang-v1")
```

```python
from tdc.single_pred import ADME
data = ADME(name="Caco2_Wang")
```

### 2.4 Harmonization (the data agent's real job)

Per (target, source) pair the data agent must:
1. **Unit reconciliation** — convert source to the target's transformed space. HLM: both → log10
   mL/min/kg (trivial). KSOL: convert Biogen µg/mL → µM via MW (`MW = Descriptors.MolWt(mol)`,
   `µM = µg/mL ÷ MW × 1000`), then log10.
2. **Calibration check** — if any compounds overlap (canonical SMILES / InChIKey), regress source vs
   target to detect a systematic offset/scale difference between labs.
3. **Merge strategy decision** — default to **co-training the source as a separate auxiliary task/head**
   (robust to inter-lab calibration offsets), *not* pooling into the same target column. Only pool
   into one column if the calibration check shows they're aligned. This decision is logged and is one
   of the genuinely agentic choices.
4. **Standardization** — canonical SMILES, salt strip, neutralize (RDKit / `chembl_structure_pipeline`),
   dedupe, drop the flagged bad slices (§2.2) per toggle.
5. **Pool hygiene (required for Chemprop MT).** After merging intra-task and external sources, **drop
   any row that has no label in any task column** before handing the pool to the ML agent. Sparse
   arms (`baseline`, `external`) otherwise retain thousands of all-NaN ExpansionRx rows; with
   Chemprop's masked multitask loss and `batch_size=64`, many training batches then have zero valid
   labels → loss = 0/0 = NaN → all-NaN predictions (`RAE=nan`, `n_test=0`). Multi-task arms with
   intra-task auxiliaries rarely hit this because their extra heads label nearly every row. See
   [CHEMPROP_STABILITY.md](CHEMPROP_STABILITY.md) for the full diagnosis.

---

## 3. Modelling (ML agent)

Default to **Chemprop v2** multitask; the ML agent has flexibility to choose the transfer mechanism.

> Verify the current Chemprop v2.x API before coding — it's PyTorch-Lightning based and the API has
> changed across minor versions. Key capabilities to use: multitask regression with **masked loss over
> NaN targets** (a molecule contributes only to endpoints it has labels for), and transfer via loading
> a pretrained checkpoint with optional frozen MPNN encoder.

Three mechanisms, selectable per arm:
- **MT co-train** — single Chemprop MPNN, one head per task (the 9 ExpansionRx endpoints, plus any
  harmonized external endpoints as extra heads). Masked loss. This is the workhorse and yields the
  multi-target predictions the challenge scores.
- **Pretrain → fine-tune** — pretrain the MPNN on the data-rich source (e.g. Biogen HLM), then
  fine-tune on the `n` target samples; option to freeze the encoder and fit only the head (strongest at
  the smallest `n`).
- **Frozen-embedding + light head** — extract Chemprop (or fingerprint) embeddings, fit ridge/GBM head
  per task. Cheap to run across the whole sweep; good few-shot baseline.

**Mandatory strong baseline: LightGBM** (or XGBoost) on `ECFP4 (2048-bit) + RDKit descriptors`,
single-task. ADMET benchmarks routinely show tuned GBMs beating fancier models — the transfer story
is only credible if the MT/transfer arms beat this. Make it a real, tuned baseline, not a strawman.

All models: train in transformed target space, invert predictions before scoring. Fixed seeds,
deterministic where possible.

**Chemprop MT on LUMI / SLURM (stability requirements).** The wrapper in `src/models/chemprop_mt.py`
must:

- Use **masked loss over NaN targets** (Chemprop built-in) *and* ensure the data agent never feeds
  all-unlabeled rows (§2.4 step 5).
- Set **`enable_checkpointing=False`** when many array tasks share a working directory — Lightning's
  default checkpoint versioning races under concurrency.
- Wire the FFN **`output_transform`** as chemprop's `UnscaleTransform`, not a raw sklearn scaler.
- Pin **`LightningEnvironment`** on Cray/Shasta login allocations to avoid PMI init failures.
- Optionally lower **`max_lr`** and enable **`gradient_clip_val`** on very sparse multitask pools.

Details and debugging checklist: [CHEMPROP_STABILITY.md](CHEMPROP_STABILITY.md).

---

## 4. Experiment protocol (planner-driven)

For each target endpoint, with the official `test` split fixed, sweep training size
`n ∈ {25, 50, 100, 250, 500, full}` (cap at available labels) and run four arms:

1. **Baseline** — single-task, `n` target samples only.
2. **+ intra-task** — MT trunk on `n` target samples + all available ExpansionRx source endpoints.
3. **+ external** — `n` target samples + the harmonized external source for that target.
4. **+ both** — full stack.

Per (endpoint, arm, n): **≥5 seeds** (different subsamples + init), report mean RAE with error band.
A "lift" inside the band is not a lift. Expected signature: transfer arms below baseline at small `n`,
converging as `n` grows — that shape *is* the answer to "when does it help."

Subsampling: respect temporal order via compound ID where it matters; keep the held-out test fixed
across all arms and seeds.

**Outputs per run:** tidy results table (`endpoint, arm, n, seed, rae, mae, r2, spearman`), the curve
plots, and a run manifest (configs, data versions, git SHA).

---

## 5. Agent architecture and contracts

Three agents over a deterministic toolset. The agents are **LLM-driven controllers that call typed
tools**; the heavy lifting (train, eval, harmonize) is deterministic Python. Build the tools first,
then the agents.

```
Planner agent ── dispatches ──▶ Data agent ──(training pool)──▶ ML agent ──(RAE)──▶ results
     ▲                                                                                  │
     └──────────────────────── reads results, picks next arm ───────────────────────────┘
```

Use plain dataclasses (or pydantic) for the contracts so an agent can be a deterministic function
*or* an LLM controller behind the same interface.

### 5.1 Data agent

**Responsibility:** given a target endpoint and requested pool size, return a curated, harmonized
training pool. Decides source selection, unit reconciliation, merge strategy, bad-slice exclusion.

```python
@dataclass
class PoolRequest:
    target_endpoint: str            # e.g. "HLM CLint"
    n: int | None                   # target-label budget; None = all
    include_intra_task: bool        # add ExpansionRx source endpoints
    include_external: bool          # add harmonized external source(s)
    seed: int
    exclude_flagged_slices: bool = True

@dataclass
class TrainingPool:
    df: "pd.DataFrame"              # SMILES + one column per (transformed) task
    task_columns: list[str]         # target first, then auxiliaries
    target_endpoint: str
    provenance: dict                # source ids, versions, n per task, transforms applied
    harmonization_log: dict         # unit conversions, calibration checks, merge decisions
```

### 5.2 ML agent

**Responsibility:** given a training pool and the fixed test set, choose a mechanism, train, evaluate,
return metrics. Has flexibility over mechanism/featurizer/hyperparameters.

```python
@dataclass
class TrainRequest:
    pool: TrainingPool
    mechanism: Literal["mt_cotrain", "pretrain_finetune", "frozen_embed", "gbm_baseline"]
    test_df: "pd.DataFrame"         # official test split, target labels held for scoring only
    seed: int

@dataclass
class EvalResult:
    target_endpoint: str
    arm: str
    n: int
    seed: int
    rae: float                      # on native scale, per challenge definition
    extra_metrics: dict             # mae, rmse, r2, spearman
    model_manifest: dict            # mechanism, hyperparams, checkpoint path
```

### 5.3 Planner agent

**Responsibility:** owns the sweep. Picks target/source pairings from the coverage profile, enumerates
(arm, n, seed) jobs, dispatches to data+ML agents, collects `EvalResult`s, and writes the
characterization report. The "explain when transfer helps" deliverable is the planner's output, not an
afterthought.

```python
@dataclass
class ExperimentPlan:
    target_endpoints: list[str]
    arms: list[str]
    n_grid: list[int | None]
    seeds: list[int]

# planner.run(plan) -> ResultsTable + curves + report.md
```

Agentic decisions worth making real (so the agent layer isn't theater): data agent choosing/ skipping a
noisy source and choosing merge strategy from the calibration check; ML agent choosing mechanism from
pool shape (tiny `n` → frozen-embed/freeze-encoder); planner choosing which pairings to explore next
based on observed lift and writing the interpretation.

---

## 6. Repo layout

```
.
├── PLAN.md
├── pyproject.toml
├── README.md
├── conf/                       # hydra/yaml configs: endpoints, n_grid, seeds, sources
├── src/
│   ├── data/
│   │   ├── expansionrx.py       # load, profile coverage, transforms, slice flags
│   │   ├── registry.py          # external source registry + endpoint mapping
│   │   ├── harmonize.py         # unit reconciliation, calibration, merge strategy
│   │   └── standardize.py       # SMILES canonicalize/strip/neutralize/dedupe
│   ├── models/
│   │   ├── chemprop_mt.py        # MT co-train + pretrain/finetune wrappers
│   │   ├── frozen_embed.py
│   │   └── gbm_baseline.py
│   ├── eval/
│   │   ├── metrics.py            # RAE/MA-RAE (pluggable) + diagnostics
│   │   └── curves.py             # learning-curve plotting
│   ├── agents/
│   │   ├── contracts.py          # dataclasses above
│   │   ├── data_agent.py
│   │   ├── ml_agent.py
│   │   └── planner.py
│   └── experiment/
│       └── runner.py             # job enumeration, seeding, manifests
├── tests/
└── results/                    # tidy tables, plots, run manifests, report.md
```

---

## 7. Build sequence (milestones)

**M0 — scaffold + data.** Repo, env, load ExpansionRx, coverage profile, transforms, standardization,
slice flags. Test: counts and ranges match §2.1.

**M1 — metric + GBM baseline + first curve.** Implement RAE/MA-RAE (verified against official scoring).
LightGBM single-task on `HLM CLint`. Produce the baseline `n`-vs-RAE curve with seeds. *This is the
end-to-end skeleton — everything hangs off it.*

**M2 — Chemprop MT + intra-task arm.** MT co-train across ExpansionRx endpoints; masked NaN loss.
Add the `+ intra-task` arm for HLM. Confirm it beats (or doesn't) the GBM honestly.

**M3 — external transfer for HLM.** Wire Polaris `biogen/adme-fang-v1`, harmonize HLM (log10 mL/min/kg),
calibration check, co-train as auxiliary head. Add `+ external` and `+ both` arms. **First complete
transfer story.**

**M4 — contrast endpoint.** Repeat M2–M3 for `MBPB` (no external source → intra-task only) or
`Caco-2 Papp A>B` (TDC `Caco2_Wang`). Show the contrast in match quality.

**M5 — wrap agents.** Put the data/ML/planner controllers around the deterministic tools using the
§5 contracts. Planner runs the full sweep and emits `report.md`.

**M6 — stretch.** ASAP antiviral source; pretrain/finetune + frozen-embed mechanisms; all 9 endpoints;
LLM-in-the-loop decisions for the agents; submit to the (now-open) leaderboard for an external check.

---

## 8. First-target spec — HLM CLint (do this first, concretely)

- **Target column:** `HLM CLint` (mL/min/kg) → `log10`. Floor zeros/near-zeros at a small value first.
- **Intra-task auxiliaries:** all other ExpansionRx endpoints (`MLM CLint` especially — closely related
  rodent analog), transformed per §2.1, as extra heads with masked loss.
- **External source:** Biogen/Fang `LOG HLM_CLint (mL/min/kg)` from `biogen/adme-fang-v1`. Already
  log10 mL/min/kg → directly comparable.
  - Calibration check: InChIKey overlap with ExpansionRx; regress to detect lab offset.
  - Merge: default **auxiliary task/head** (separate Biogen-HLM head sharing the MPNN trunk), not pooled
    into the ExpansionRx HLM column, unless calibration says aligned.
- **Arms:** baseline (LightGBM + single-task Chemprop), + intra-task, + external, + both.
- **Sweep:** `n ∈ {25,50,100,250,500,full}` of ExpansionRx HLM labels, 5 seeds.
- **Expect:** external arm gives the cleanest lift at small `n` (near-identical assay); intra-task adds
  from MLM. Quantify the gap and where it closes.

---

## 9. Guardrails

- Test split touched once, at scoring, ever. Tune on an internal `train` fold only.
- Report error bands; never claim a lift inside the noise.
- Beat the tuned GBM or say so plainly.
- Log provenance + harmonization decisions for every pool (reproducibility + the agentic-decision
  narrative).
- Verify before trusting: the RAE definition (official scoring), the Chemprop v2 API, and the Polaris
  client API are the three things most likely to have drifted — check current docs/source.
- External data is co-trained, not blindly pooled, unless a calibration check justifies pooling.

---

## 10. Environment

```
python >= 3.10
chemprop >= 2.0            # verify current API
rdkit
lightgbm                  # or xgboost
scikit-learn
datasets                  # HuggingFace, for ExpansionRx
polaris-lib               # Polaris hub (verify package/import name)
PyTDC                     # TDC external sources
pandas, numpy, scipy
matplotlib                # curves
hydra-core / pydantic     # configs + contracts
pytest
```
