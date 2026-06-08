# RUNBOOK — full 5-endpoint experiment (`results-fullexp/`)

Hands-on guide for launching, monitoring, and re-running the full ADMET transfer study on LUMI.
Companion docs: [`EXPERIMENT.md`](./EXPERIMENT.md) (design + documented findings),
[`../Readme.md`](../Readme.md) (project overview), [`../Plan.md`](../Plan.md) (build spec).

For follow-on mechanism experiments (pretrain/finetune, persisted preds), see
[`../further-improvements/RUNBOOK.md`](../further-improvements/RUNBOOK.md).

---

## 0. Golden rules (read once)

1. **Never train on a login node.** Chemprop training goes through `sbatch` to the `small-g` GPU
   partition. Manifest generation and `collect` are light but still run inside the container.
2. **Run from a LUMI login node** with `sbatch` available. The launcher exits in seconds; SLURM
   runs everything detached.
3. **This study uses the LLM agent** (`--llm` on every sweep). You need a valid Aitta token; agent
   decisions are cached under `results/llm_cache/` so repeats are cheap.
4. **Output is isolated** in `results-fullexp/` — it never touches `results/` or `results-rounds/`.
5. **Re-running the same round is safe:** finished jobs are skipped when
   `results-fullexp/<round>/runs/<job_id>.json` already exists.

---

## 1. Fixed paths (this cluster)

```bash
WORK=/pfs/lustrep1/scratch/project_462001520/Team3/Team3
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-latest.sif
VENV=/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate
ACCOUNT=project_462001520
EXP_DIR=results-fullexp
```

All commands below assume `cd "$WORK"` unless noted.

---

## 2. Experiment at a glance

Five endpoint rounds, submitted in parallel by `scripts/run_fullexp.sh`:

| Round key | Endpoint | Plan config | Results dir |
|---|---|---|---|
| `hlm` | HLM CLint | `conf/fullexp-hlm.yaml` | `results-fullexp/hlm/` |
| `mbpb` | MBPB | `conf/fullexp-mbpb.yaml` | `results-fullexp/mbpb/` |
| `ksol` | KSOL | `conf/fullexp-ksol.yaml` | `results-fullexp/ksol/` |
| `mppb` | MPPB | `conf/fullexp-mppb.yaml` | `results-fullexp/mppb/` |
| `caco2eff` | Caco-2 Permeability Efflux | `conf/fullexp-caco2eff.yaml` | `results-fullexp/caco2eff/` |

Each round: **4 arms** × **6 n values** (25, 50, 100, 250, 500, full) × **5 seeds** = **120 jobs**.
Total: **600 training jobs** + **5 collect jobs**.

Arms: `baseline` (LightGBM), `intra_task`, `external`, `both` (Chemprop v2 multitask via
`mt_cotrain`). SLURM sizing: `conf/slurm-fullexp.yaml` — 36 array tasks per round striding 4 jobs,
`array_concurrency: 100` → **185 submitted jobs** (5×36 sweeps + 5 collect), under LUMI's ~200 cap.

---

## 3. One-time setup

```bash
# Aitta token (24 h TTL): https://aitta-auth.csc.fi/myToken
echo 'PASTE_YOUR_TOKEN_HERE' > conf/.aitta_token && chmod 600 conf/.aitta_token
# or: export AITTA_API_TOKEN=...
```

Optional container shell for light local commands (collect without SLURM, debugging):

```bash
module use /appl/local/laifs/modules && module load lumi-aif-singularity-bindings
csh() {
  singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 --pwd "$WORK" "$CONTAINER" \
    bash -lc "source '$VENV' && $*"
}
csh 'python -c "import pandas, chemprop; print(\"env OK\")"'
```

---

## 4. Workflow A — launch the full study (recommended)

Run from the project root on a login node:

```bash
# 1) Preflight — Aitta + plan enumeration, no jobs
bash scripts/run_agent.sh --check

# 2) Preview sbatch scripts (submits nothing)
bash scripts/run_fullexp.sh --dry-run

# 3) Submit all 5 rounds (disconnection-safe)
nohup bash scripts/run_fullexp.sh &>results-fullexp/submit.log &
# or foreground:  bash scripts/run_fullexp.sh
```

What `run_fullexp.sh` does for each round:

1. Calls `submit_sweep.sh --llm` with the round's plan and `conf/slurm-fullexp.yaml`.
2. Chains a collect job (`scripts/collect_llm.sbatch`) with `--dependency=afterany:<sweep_id>`.
3. Appends sweep/collect IDs to `results-fullexp/JOBIDS.txt`.

You can log out immediately after step 3; reports and plots are written when each collect finishes.

---

## 5. Workflow B — submit one round only

Useful for debugging or re-running a single endpoint:

```bash
# Example: MBPB only
PLAN=conf/fullexp-mbpb.yaml \
SLURM_YAML=conf/slurm-fullexp.yaml \
RESULTS_DIR=results-fullexp/mbpb \
JOB_NAME=fx-mbpb \
  bash scripts/submit_sweep.sh --llm

# Then chain collect manually (replace SWEEP_ID with the id printed above):
SWEEP_ID=<id>
sbatch --parsable --partition=small-g \
  --dependency=afterany:${SWEEP_ID} \
  --export=ALL,RESULTS_DIR=${WORK}/results-fullexp/mbpb,AITTA_API_TOKEN \
  --chdir=${WORK} \
  --output=${WORK}/results-fullexp/mbpb/logs/collect-%j.out \
  --error=${WORK}/results-fullexp/mbpb/logs/collect-%j.err \
  scripts/collect_llm.sbatch
```

Or use the helper (expects `runs/` to exist; re-aggregates only):

```bash
RESULTS_DIR=results-fullexp/mbpb bash scripts/submit_collect.sh
```

---

## 6. Monitor progress

```bash
# Queue
squeue -u $USER
squeue -u $USER -h -o '%T %j' | sort | uniq -c   # counts by state

# Job IDs from last full launch
cat results-fullexp/JOBIDS.txt

# Per-round progress (repeat for hlm, ksol, mppb, caco2eff)
ROUND=results-fullexp/mbpb
echo "done: $(ls $ROUND/runs/*.json 2>/dev/null | wc -l) / $(wc -l < $ROUND/manifest.jsonl)"
grep -rlE "Traceback|Error" $ROUND/logs/*.err 2>/dev/null | wc -l   # failures (want 0)

# Generic helper (defaults to results/ — point at a fullexp round manually):
# TOTAL=$(wc -l < results-fullexp/mbpb/manifest.jsonl)
# DONE=$(ls results-fullexp/mbpb/runs/*.json | wc -l)

# Post-mortem on one job
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

When a round's collect job finishes, read:

```bash
cat results-fullexp/<round>/report.md
```

Round keys: `hlm`, `mbpb`, `ksol`, `mppb`, `caco2eff`.

---

## 7. Re-run / resume

| Goal | Command |
|---|---|
| Re-submit everything (skip finished jobs) | `bash scripts/run_fullexp.sh` |
| Force one job to re-train | Delete its `runs/<job_id>.json`, re-submit the round's sweep |
| Re-aggregate plots/report only (no GPU training) | `RESULTS_DIR=results-fullexp/<round> bash scripts/submit_collect.sh` |
| Fresh output tree (don't overwrite) | Copy or rename `results-fullexp/` first, or change `EXP_DIR` in `run_fullexp.sh` |

Finished jobs are detected in `src/experiment/runner.py` (`skip_if_exists` when
`runs/<job_id>.json` is present). LLM cache hits avoid re-calling Aitta for identical decisions.

---

## 8. Alternative metrics (no re-training)

Every run stores `rae`, `mae`, `rmse`, `r2`, `spearman` in `results.parquet`. Regenerate plots
and narrative for any metric (writes `curves_<metric>.png`, `ma_<metric>.png`, `report_<metric>.md`;
default RAE outputs are preserved):

```bash
# SLURM + LLM narrative
RESULTS_DIR=results-fullexp/mppb METRIC=r2 bash scripts/submit_collect.sh

# Local (inside container)
csh "python -m experiment.cli collect --results-dir results-fullexp/mppb --metric r2"
# metrics: rae | mae | rmse | r2 | spearman
```

Add `--llm` to `collect` for the agent-written report (needs token).

---

## 9. Output layout (per round)

```
results-fullexp/<round>/
├── manifest.jsonl              # one line per job
├── fx-<round>.sbatch           # generated array script
├── fx-<round>.worker.sh        # per-task worker
├── logs/
│   ├── fx-<round>-<arrayid>_<task>.{out,err}
│   └── collect-<jobid>.{out,err}
├── runs/<job_id>.json          # per-job metrics + metadata
├── results.parquet             # aggregated table (after collect)
├── curves.png , ma_rae.png     # default RAE plots
├── curves_<metric>.png         # optional metric-specific plots
└── report.md                   # LLM narrative (after collect --llm)
```

Top-level: `results-fullexp/JOBIDS.txt`, `results-fullexp/submit.log` (if launched via `nohup`).

---

## 10. Config reference

Edit before re-launching if you need to change scope:

| File | Purpose |
|---|---|
| `conf/fullexp-{hlm,mbpb,ksol,mppb,caco2eff}.yaml` | Per-round arms, n grid, seeds, mechanisms |
| `conf/slurm-fullexp.yaml` | Account, partition, time, array concurrency, binds |
| `scripts/run_fullexp.sh` | Orchestrates all 5 rounds |
| `scripts/submit_sweep.sh` | Build manifest + submit one sweep |
| `scripts/submit_collect.sh` | Submit collect-only job |

Plan YAML schema (all rounds follow this pattern):

```yaml
target_endpoints: ["MBPB"]
arms: ["baseline", "intra_task", "external", "both"]
n_grid: [25, 50, 100, 250, 500, null]   # null = full data
seeds: [0, 1, 2, 3, 4]
mechanism_by_arm:
  baseline: "gbm_baseline"
  intra_task: "mt_cotrain"
  external: "mt_cotrain"
  both: "mt_cotrain"
exclude_flagged_slices: true
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: no Aitta token` | Missing/expired token | Refresh `conf/.aitta_token` from https://aitta-auth.csc.fi/myToken |
| `ERROR: sbatch not found` | Not on a login node | SSH to LUMI login; don't run launcher from a compute node |
| `sbatch` rejected / bad account | Wrong SLURM account | Use `project_462001520` (set in `conf/slurm-fullexp.yaml`) |
| Collect job fails, `runs/` incomplete | Sweep still running or failed tasks | Check `logs/fx-*-*.err`; fix failures, delete bad `runs/*.json`, re-submit sweep |
| Job exits instantly, no `runs/*.json` | Array task error | Read the matching `logs/fx-<round>-<id>_<task>.err` |
| Re-run retrains everything | Deleted `runs/` or new results dir | Expected; only existing `runs/*.json` files are skipped |
| `ModuleNotFoundError: pandas` on login node | Ran Python outside container | Use `csh '...'` or the SLURM scripts (they wrap the container) |

Logs to inspect first: `results-fullexp/<round>/logs/fx-<round>-*_<task>.err`.

---

## 12. Quick recipes

```bash
# A) Full study from scratch (standard path)
bash scripts/run_agent.sh --check
nohup bash scripts/run_fullexp.sh &>results-fullexp/submit.log &

# B) Dry-run only — verify plans render
bash scripts/run_fullexp.sh --dry-run

# C) Re-run one round after fixing a failed job
rm results-fullexp/mbpb/runs/MBPB__external__100__s2.json   # example
PLAN=conf/fullexp-mbpb.yaml SLURM_YAML=conf/slurm-fullexp.yaml \
RESULTS_DIR=results-fullexp/mbpb JOB_NAME=fx-mbpb \
  bash scripts/submit_sweep.sh --llm
RESULTS_DIR=results-fullexp/mbpb bash scripts/submit_collect.sh

# D) Regenerate R² plots + report for all rounds (after training done)
for r in hlm mbpb ksol mppb caco2eff; do
  RESULTS_DIR=results-fullexp/$r METRIC=r2 bash scripts/submit_collect.sh
done
```
