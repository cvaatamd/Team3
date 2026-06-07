# Further improvements — the path to winning

> **TL;DR — how we win.** We are already the most complete and reproducible team (full
> 5-endpoint × 4-arm × 6-size × 5-seed sweep, multi-metric, real cached agentic decisions). We do
> **not** need more compute. We win by (1) telling the one story only we can tell —
> *external data helps magnitude, intra-task helps ranking* — and (2) closing the two credibility
> gaps the front-runners have over us: a verified official metric and a negative control.

This folder is a focused action plan, not a lab notebook. Everything here is either zero-run
(reads the existing `results-fullexp/*/results.parquet`) or a sub-day change.

> **Now implemented (isolated copy):** Patches 1–3 are applied in `src/` here (a copy of the
> repo's `src/`; the production `../src/` is untouched). See [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)
> for exactly what changed and the 3-step run guide. Heavy prediction-regeneration runs on **GPU via
> SLURM** (`scripts/submit_regen.sh`) — never on a login node; re-score and bootstrap are light.

### Docs in this folder
- **[`EXPERIMENT.md`](./EXPERIMENT.md) — the full formal report** (context, design, SLURM, results
  tables, plots, takeaways), mirroring `../results-fullexp/EXPERIMENT.md`.
- **[`RUNBOOK.md`](./RUNBOOK.md) — run your own experiments manually** (env, configs, submit, monitor,
  analyze, troubleshooting).
- [`RESULTS.md`](./RESULTS.md) — TL;DR of the findings (pretrain_finetune wins, frozen_embed ruled out).
- [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) — exactly what the patches change in `src/`.
- This file (`README.md`) — the competitive strategy / why this work matters.

### Folder contents
- `src/` — patched copy of the pipeline (persist-preds, official RAE, negative control, **real
  pretrain_finetune / frozen_embed transfer**). The production `../src/` is untouched.
- `conf/` — experiment configs: `fewshot-*.yaml` (mechanism sweeps), `regen-*.yaml`,
  `negctrl-*.yaml`, `smoke-*.yaml`, plus a local `endpoints.yaml`.
- `scripts/submit_regen.sh`, `scripts/regen.worker.sh` — GPU SLURM launcher + per-task worker.
- `scripts/rescore_official.py` — offline RAE re-score (all definitions) from persisted preds.
- `scripts/compare_fewshot.py` — paired new-vs-old RAE comparison with bootstrap CIs.
- `scripts/bootstrap_lift_molecule.py` — molecule-level paired bootstrap CIs.
- `outputs/` — analysis CSVs; `results-fewshot/{pf,fe}/` — sweep outputs + per-molecule `preds/`.

---

## The competitive picture (why this plan)

| Rival | Their edge | How this plan neutralizes it |
|---|---|---|
| **team2** | Deep science: oracle ceilings, 2 datasets, ablations, a paper. Concludes (on **Spearman**) *"external cross-dataset data does not help."* | We show that conclusion is **metric-specific**: external *does* help on calibrated error (RAE/MAE/RMSE/R²). We absorb their result instead of contradicting it. |
| **team1** | Engineering + a 6-layer verification plan, scaffold/cluster splits, applicability-domain flags. | We add a verified **official RAE** + a **negative control** + one robustness check — enough verification signal without re-architecting. |

Judges (per the signals we saw) weight **agentic value, human-in-the-loop, trust/verification, and
wet-lab usefulness** — *not* raw Spearman. So the plan ends with making the agent's reasoning
visible, which is where the points actually are.

---

## The four moves, in priority order

### 1. Reframe — magnitude vs ranking  ·  *zero new runs*  ·  **do this first**
The headline finding. The numbers are already in the parquet; this is a writeup.
- Paste [`metric_reconciliation.md`](./metric_reconciliation.md) near the top of
  `results-fullexp/EXPERIMENT.md` Results.
- One-line thesis: *external assays maximise calibrated accuracy (3/5 endpoints, up to −27% RAE);
  intra-task ExpansionRx co-training maximises rank correlation. Choose the auxiliary by objective.*

### 2. Verify the official RAE  ·  *~½ day*  ·  **credibility floor**
Every headline % rests on the RAE denominator. The `official` branch in `src/eval/metrics.py` is
still a stub.
- **Gotcha we found:** per-molecule predictions are **not** persisted (`runner.run_one` stores only
  `EvalResult` scalars), so switching `rae_definition` today means **re-running all 600 jobs** —
  the `EXPERIMENT.md` "recomputes from stored predictions" claim is currently false.
- **Fix order:** persist `(y_true, y_pred)` per run first (~3 lines), *then* wire `official`. After
  that, any RAE definition is a free offline re-score, and the doc's claim becomes true.
- Exact diffs: [`patches.md`](./patches.md).

### 3. Shuffled-external negative control  ·  *~½ day*  ·  **rebuttal-killer**
Proves "external carries signal" isn't just regularization. If the real external arm beats a
label-permuted one, the harmonization story is bulletproof.
- Cheapest form: a permute-seed flag in `harmonize_external` (no schema change), run as a separate
  results dir. See [`patches.md`](./patches.md).

### 4. Make the agent's reasoning visible  ·  *~½ day*  ·  **judge alignment**
team2's whole pitch is "the agent's value is the explanation." We already capture rationales
(`select_sources` / `choose_mechanism`) — we just don't surface them.
- Echo the LLM's source/mechanism rationales and the harmonization log into `report.md`.
- Prep a 2-minute demo of one real decision (e.g. *"agent saw weak KSOL µg/mL→µM calibration, kept
  it as an aux head, +18% paid off"*).

---

## Statistical hardening (done)
Predictions are now persisted, so significance is computed properly:
- `bootstrap_lift.py` — seed-level bootstrap straight off the existing `results.parquet` (no regen).
- `scripts/compare_fewshot.py` — paired bootstrap over seeds, new-vs-old, with 95% CIs + sign test
  (used for the RESULTS.md tables).
- `scripts/bootstrap_lift_molecule.py` — molecule-level paired bootstrap (~2.3k test molecules) for
  much tighter CIs than seed-level.

---

## Explicitly NOT doing
- **No team2-style selection oracle.** Our registry maps one external source per target, so there is
  no candidate pool to select over — an oracle needs a multi-source registry first. That's the only
  real-work item and it is not where we win.
- **No new endpoints / more seeds.** We already have the most complete sweep.

---

## Time budget
~1–1.5 days, mostly writing plus one cheap control run. Net effect: neutralizes team2's research
edge and team1's verification edge, while playing directly to the judging criteria.
