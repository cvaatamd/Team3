# Implementation — Patches 1–3 (isolated copy)

All changes live in **this folder only**. `further-improvements/src/` is a full copy of the repo's
`src/` with the three patches applied; the production `../src/` is **untouched**. Run the copy by
putting it first on `PYTHONPATH` (the SLURM worker does this automatically).

> **Login-node rule (enforced):** no intensive CPU/training on a login node. The only heavy step —
> regenerating per-molecule predictions (Chemprop training) — runs on **GPU compute nodes via
> SLURM** (`submit_regen.sh` → array job). Everything else here (manifest build, re-score,
> bootstrap) is light I/O and runs in seconds inside the container.

---

## What changed, and where (in `further-improvements/src/`)

### Patch 1 — persist per-molecule predictions
- `agents/ml_agent.py`
  - `TrainSpec` gains `preds_dir` + `job_id`.
  - `MLAgent.run` calls new `MLAgent._persist_preds(...)`, which writes native-scale
    `(y_true, y_pred)` + a `.meta.parquet` (endpoint/arm/n/seed/mechanism/test_range) per run.
- `experiment/runner.py::run_one` — sets `preds_dir=<results_dir>/preds` and `job_id` on the spec,
  and runs with `--force` semantics so a regen overwrites stale scalar-only runs.
- `agents/planner.py::in_process_dispatcher` — optional `preds_dir` for local runs.

**Effect:** any RAE definition and molecule-level statistics become free offline operations.
This also makes the EXPERIMENT.md claim ("recompute from stored predictions") actually true —
it was false before (only scalars were saved).

### Patch 2 — official RAE
- `eval/metrics.py::rae` — the `official` branch is implemented (range-normalized MAE, matching
  the challenge's headline definition) instead of raising `NotImplementedError`. It carries an
  explicit **VERIFY** note: diff against the HF Space scoring code and adjust the denominator if
  the official metric uses log space / a robust range / fixed published ranges.

**Effect:** `rescore_official.py` recomputes `range_normalized`, `vs_mean`, and `official` from the
persisted preds and prints `max |official - range_normalized|` so you can show they agree.

### Patch 3 — shuffled-external negative control
- `data/harmonize.py::harmonize_external` — new `shuffle_seed`; when set, the harmonized source
  labels are permuted across molecules (marginal preserved, structure↔activity link destroyed).
  Logged as `NEGATIVE_CONTROL=shuffled(seed=...)` in the harmonization notes.
- Threaded through: `agents/contracts.py` (`PoolRequest`, `JobSpec`, `ExperimentPlan` +
  `enumerate_jobs`), `agents/planner.py::_pool_request_from_job`, `agents/data_agent.py::build_pool`,
  `experiment/runner.py` (`load_plan`, `plan_from_manifest`).
- Driven by a single YAML key: `shuffle_external_seed`.

**Effect:** running the `external`/`both` arms with `shuffle_external_seed` set produces a control;
if the real external arm beats the shuffled one, the lift is genuine transfer, not regularization.

### Patch 4 — real `pretrain_finetune` / `frozen_embed` transfer
- `agents/ml_agent.py` — new `MLAgent._pretrain_aux_checkpoint(spec)`: pretrains the Chemprop encoder
  on the auxiliary heads (full multitask head on aux-labeled rows, so the checkpoint reloads without
  a task-dimension mismatch), saves it to `<results_dir>/pretrain_ckpts/`, returns the path.
  - `pretrain_finetune` now loads that checkpoint and fine-tunes on the target (was previously a
    silent no-op: it read a `pretrained_checkpoint` from provenance that nothing ever set).
  - `frozen_embed` now passes the checkpoint to the ridge-on-embeddings head (was fingerprint-only).
- **Effect:** the two "advanced" mechanisms in `mechanism_by_arm` finally do real transfer. See
  [`RESULTS.md`](./RESULTS.md): `pretrain_finetune` significantly lowers few-shot RAE in 3/12 cells
  and resolves the n=25 collapse; `frozen_embed` is worse (ruled out). Analysis via
  `scripts/compare_fewshot.py`.

> **Full manual run guide:** [`RUNBOOK.md`](./RUNBOOK.md). The 3-step quickstart below is for the
> Patch-1/2/3 regen/control flow specifically.

---

## How to run (3 steps)

All commands assume project root `/pfs/lustrep1/scratch/project_462001520/Team3/Team3`.

### 1. Regenerate predictions on GPU (SLURM — never on a login node)
```bash
# headline endpoints (persists preds + scalars). Repeat per endpoint, or loop.
for ep in mbpb ksol mppb; do
  bash further-improvements/scripts/submit_regen.sh \
      further-improvements/conf/regen-$ep.yaml \
      $PWD/results-fullexp-regen/$ep
done

# negative control (shuffled external) — the "one cheap re-run of two arms"
for ep in mbpb ksol mppb; do
  bash further-improvements/scripts/submit_regen.sh \
      further-improvements/conf/negctrl-$ep.yaml \
      $PWD/results-fullexp-negctrl/$ep
done
```
`submit_regen.sh` builds the manifest in-container (fast, no GPU) then `sbatch`es the array. Monitor
with `squeue -u $USER`.

### 2. Confirm headline numbers (official RAE) — light, after step 1
```bash
singularity exec --bind /scratch/project_462001520:/scratch/project_462001520 \
  --bind /pfs/lustrep1:/pfs/lustrep1 --pwd $PWD \
  /appl/local/laifs/containers/lumi-multitorch-latest.sif \
  bash -lc "source /scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate && \
            PYTHONPATH=$PWD/further-improvements/src \
            python further-improvements/scripts/rescore_official.py \
              --preds-dir results-fullexp-regen/mbpb/preds \
              --preds-dir results-fullexp-regen/ksol/preds \
              --preds-dir results-fullexp-regen/mppb/preds"
```
Expect `official == range_normalized` (max abs diff ~0) → the EXPERIMENT.md percentages are on the
challenge's headline metric.

### 3. Molecule-level bootstrap CIs — light, after step 1
```bash
# inside the same container/venv (has pandas/scipy):
python further-improvements/scripts/bootstrap_lift_molecule.py \
  --preds-dir results-fullexp-regen/mbpb/preds \
  --preds-dir results-fullexp-regen/ksol/preds \
  --preds-dir results-fullexp-regen/mppb/preds
```
Tight, properly-paired CIs over the ~2.3k test molecules (vs seed-level CIs from
`compare_fewshot.py`). Use to state "external beats baseline on RAE, 95% CI excludes 0."

### Negative-control comparison
After both step-1 sweeps finish, compare real vs shuffled external RAE (e.g. with `rescore_official.py`
on each preds dir, or a small join): the real external arm should be clearly better. That gap is the
proof the harmonized signal is real.

---

## Notes
- The regen confs trim `n_grid` to `[50, null]` (few-shot + full — the two reported regimes) to save
  GPU time; widen to the full `[25, 50, 100, 250, 500, null]` for an exact reproduction.
- The copy keeps the LLM path intact (`--llm`), so the agentic source/mechanism decisions are
  exercised during regen exactly as in the original sweep (decisions are cached on disk).
- Verified: editor linter clean; bash `-n` clean. (The login node's `python3` is 3.6 and cannot
  import the stack — all Python here targets the 3.10+ container interpreter.)
