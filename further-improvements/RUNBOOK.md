# RUNBOOK — run your own experiments manually

Hands-on guide for launching, monitoring, and analyzing experiments with the patched pipeline in
this folder. Everything is verified against the runs that produced [`RESULTS.md`](./RESULTS.md).
Companion docs: [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) (what the patches change),
[`RESULTS.md`](./RESULTS.md) (findings), [`README.md`](./README.md) (strategy).

---

## 0. Golden rules (read once)

1. **Never train on a login node.** Heavy work (Chemprop training) goes through `sbatch` to the
   `small-g` GPU partition. Manifest builds, re-scoring, and bootstraps are light and run in the
   container in seconds.
2. **Use the patched code by *prepending* it to `PYTHONPATH` inside the container.** Do **not**
   `export SINGULARITYENV_PYTHONPATH` — that clobbers the container's own `PYTHONPATH` and hides
   `pandas`/`torch` (this bit us; see Troubleshooting).
3. **For controlled mechanism experiments, disable the LLM** (`ADMET_USE_LLM=0`). Otherwise the LLM
   agent may override the mechanism you asked for. Leave it on (`=1`) only if you *want* the agent to
   choose sources/mechanisms (needs `conf/.aitta_token`).

## 1. Fixed paths (this cluster)

```bash
WORK=/pfs/lustrep1/scratch/project_462001520/Team3/Team3
CODE=$WORK/further-improvements/src                              # the PATCHED copy
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-latest.sif
VENV=/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate
ACCOUNT=project_462001520                                        # SLURM compute account
```

The patched tree resolves `conf/endpoints.yaml` and `data_cache/` relative to itself; both are
already in place (`further-improvements/conf/endpoints.yaml`, `further-improvements/data_cache ->
../data_cache`). If you copy this folder elsewhere, recreate them.

## 2. Drop into the working environment (light commands only)

A reusable container shell for any light Python (manifest build, re-score, analysis):

```bash
module use /appl/local/laifs/modules && module load lumi-aif-singularity-bindings
csh() {  # "container shell": run a command inside the container venv with patched code first
  singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 --pwd "$WORK" "$CONTAINER" \
    bash -lc "source '$VENV' && export PYTHONPATH='$CODE':\${PYTHONPATH:-} && $*"
}
# smoke check:
csh 'python -c "import pandas, chemprop, sklearn; print(\"env OK\")"'
```

---

## 3. Write an experiment config

A config is a small YAML. Full schema (all keys):

```yaml
target_endpoints:            # one or more; exact strings below
  - "MBPB"
arms: ["baseline", "intra_task", "external", "both"]   # any subset
n_grid: [25, 50, 100, 250, 500, null]   # target-label budgets; null = full data
seeds: [0, 1, 2, 3, 4]
mechanism_by_arm:            # which model each arm uses
  baseline:   "gbm_baseline"
  intra_task: "mt_cotrain"
  external:   "pretrain_finetune"
  both:       "pretrain_finetune"
exclude_flagged_slices: true # drop applicability-domain-flagged test rows
shuffle_external_seed: 1234  # OPTIONAL: permute external labels = negative control
```

**Valid `target_endpoints`** (must match exactly, incl. spaces/case):
`"HLM CLint"`, `"MBPB"`, `"KSOL"`, `"MPPB"`, `"Caco-2 Permeability Efflux"`
(others defined in `conf/endpoints.yaml`: `"LogD"`, `"MLM CLint"`,
`"Caco-2 Permeability Papp A>B"`, `"MGMB"`).

**Valid `arms`:** `baseline` (target only), `intra_task` (+ ExpansionRx aux endpoints),
`external` (+ harmonized Biogen source), `both`.

**Valid mechanisms** (`mechanism_by_arm` values):

| mechanism | what it does | notes |
|---|---|---|
| `gbm_baseline` | LightGBM on ECFP4+descriptors, single-task | for `baseline`; ignores aux |
| `mt_cotrain` | Chemprop v2 multitask, trained from scratch on target+aux jointly | the original sweep's transfer model |
| `pretrain_finetune` | **pretrain** Chemprop encoder on the aux heads, **then fine-tune** on target | best at few-shot; ~2× train time (two phases) |
| `frozen_embed` | pretrain encoder, **freeze** it, ridge head on its embeddings | cheap but underperformed here (see RESULTS.md) |

Put your config in `further-improvements/conf/`. Start from
`conf/fewshot-pretrain_finetune.yaml` or `conf/smoke-fewshot.yaml`.

> **Tip — always smoke first.** Make a 1-job config (one endpoint, one arm, one `n`, one seed) and
> run it before a big sweep. `conf/smoke-fewshot.yaml` (external arm) and `conf/smoke-both.yaml`
> (multi-aux `both` arm) are ready-made.

---

## 4. Workflow A — submit a full sweep (recommended)

`submit_regen.sh` builds the manifest in-container (fast, no GPU) and `sbatch`es a strided array.
Predictions are persisted to `<results_dir>/preds/`.

```bash
# usage: submit_regen.sh CONF RESULTS_DIR [CONCURRENCY=100] [MAX_TASKS=36]
ADMET_USE_LLM=0 bash $WORK/further-improvements/scripts/submit_regen.sh \
  $WORK/further-improvements/conf/fewshot-pretrain_finetune.yaml \
  $WORK/further-improvements/results-fewshot/pf
```

- It prints `manifest jobs: N` and `submitted array job <ID>`.
- `MAX_TASKS` = number of array tasks; each task strides `ceil(N/MAX_TASKS)` jobs (keeps the array
  under LUMI's submit limit). `CONCURRENCY` = max tasks running at once.
- **Gate one submission on another** (e.g. full sweep only after a smoke passes):
  ```bash
  SBATCH_DEP=<smoke_job_id> ADMET_USE_LLM=0 bash .../submit_regen.sh CONF RESULTS_DIR
  ```

The array runs `scripts/regen.worker.sh`, which loops its slice of manifest indices inside the
container. Knobs it honors: `ADMET_USE_LLM` (0/1), `AITTA_API_TOKEN_FILE`.

## 5. Workflow B — run a single job by hand

Useful for debugging one cell. Two steps: build a manifest, then run one index on a GPU.

```bash
# (a) build the manifest from a conf (light; in container):
csh "python -c \"
from pathlib import Path
from experiment.runner import load_plan, write_manifest, read_manifest
plan = load_plan(Path('further-improvements/conf/smoke-fewshot.yaml'))
write_manifest(plan, Path('/tmp/my_manifest.jsonl'))
print('jobs:', len(read_manifest(Path('/tmp/my_manifest.jsonl'))))\""

# (b) grab an interactive GPU and run index 0 (heavy -> compute node, NOT login):
salloc --account=$ACCOUNT --partition=small-g --gpus-per-node=1 \
       --cpus-per-task=7 --mem=60G --time=1:00:00
# ...once you get the node, inside the allocation:
RESULTS_DIR=$WORK/further-improvements/results-fewshot/manual \
MANIFEST=/tmp/my_manifest.jsonl N_JOBS=1 N_TASKS=1 ADMET_USE_LLM=0 \
  bash $WORK/further-improvements/scripts/regen.worker.sh 0
```

The worker handles modules, the container, `PYTHONPATH`, and the `run-job` call. Equivalent raw CLI
(if you're already inside the container shell on a GPU node):

```bash
csh "python -m experiment.cli run-job \
  --manifest /tmp/my_manifest.jsonl --index 0 \
  --results-dir $WORK/further-improvements/results-fewshot/manual --force"
# add --llm to let the agent choose source/mechanism (needs the token)
```

CLI subcommands (`python -m experiment.cli ...`): `submit-sweep`, `run-job`, `collect`.

---

## 6. Monitor a running sweep

```bash
squeue -u $USER                                  # all your jobs
squeue -u $USER -h -o '%T %j' | sort | uniq -c   # counts by state
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS  # post-mortem

D=$WORK/further-improvements/results-fewshot/pf   # your RESULTS_DIR
ls $D/preds/*.parquet | grep -v meta | wc -l       # finished jobs (preds written)
grep -rhE "done in|RAE=" $D/logs/*.err | tail       # progress lines
grep -rlE "Traceback|Error|must match" $D/logs/*.err | wc -l   # failures (want 0)
```

Logs: `<RESULTS_DIR>/logs/regen-<arrayid>_<task>.{out,err}`.
Pretrain checkpoints: `<RESULTS_DIR>/pretrain_ckpts/`.

---

## 7. Analyze (light; in the container shell)

```bash
# (1) Recompute RAE under every definition from persisted preds; confirms official == range_normalized
csh "python further-improvements/scripts/rescore_official.py \
  --preds-dir further-improvements/results-fewshot/pf/preds \
  --out further-improvements/outputs/rescore_pf.csv"

# (2) Paired new-vs-old comparison (delta = old - new; >0 means LOWER error) with bootstrap 95% CI
csh "python further-improvements/scripts/compare_fewshot.py \
  --new-preds further-improvements/results-fewshot/pf/preds --label pretrain_finetune \
  --old-results 'results-fullexp/*/results.parquet' --old-mechanism mt_cotrain \
  --out further-improvements/outputs/compare_pretrain_finetune.csv"

# (3) Molecule-level paired bootstrap CIs (tighter than seed-level)
csh "python further-improvements/scripts/bootstrap_lift_molecule.py \
  --preds-dir further-improvements/results-fewshot/pf/preds"

# (4) Aggregate into a report + learning-curve plots (per RESULTS_DIR, any metric)
csh "python -m experiment.cli collect \
  --results-dir further-improvements/results-fewshot/pf --metric rae"
```

`--preds-dir` repeats (pass several to pool rounds). Outputs land in `further-improvements/outputs/`.

### Negative control (is the external signal real?)
Run a sweep with `shuffle_external_seed:` set (permutes external labels), then compare its
`external`/`both` RAE to the unshuffled run. Real > shuffled ⇒ genuine transfer, not regularization.
Ready configs: `conf/negctrl-{mbpb,ksol,mppb}.yaml`.

---

## 8. Output layout (per RESULTS_DIR)

```
<RESULTS_DIR>/
├── manifest.jsonl                      # one line per job (endpoint/arm/n/seed/mechanism/...)
├── regen.sbatch                        # the generated array script
├── logs/regen-<A>_<task>.{out,err}     # per-task stdout/stderr
├── pretrain_ckpts/<stem>.ckpt          # encoder checkpoints (pretrain_finetune/frozen_embed)
├── preds/
│   ├── <EP>__<arm>__n<n>__s<seed>.parquet        # native-scale (y_true, y_pred)
│   └── <EP>__<arm>__n<n>__s<seed>.meta.parquet   # endpoint/arm/n/seed/mechanism/test_range
└── runs/ , results.parquet , report.md , curves*.png   # after `collect`
```

Filename stem encodes the cell, e.g. `KSOL__both__n50__s3` (`nfull` for full data).

---

## 9. Troubleshooting (errors we actually hit)

| Symptom in `*.err` | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | `SINGULARITYENV_PYTHONPATH` clobbered the container's `PYTHONPATH` | Don't export it; prepend `PYTHONPATH` *inside* the container (already fixed in `regen.worker.sh`) |
| `FileNotFoundError: .../further-improvements/conf/endpoints.yaml` | patched tree missing the endpoint config | `cp ../conf/endpoints.yaml conf/` (already done) |
| external/aux data re-downloading / missing | `data_cache/` not found in patched tree | `ln -sfn ../data_cache further-improvements/data_cache` (already done) |
| `RuntimeError: size of tensor a (10) must match b (9)` | pretrain vs fine-tune task counts differed | fixed: pretrain now uses the full multitask head on aux-labeled rows |
| job exits in ~1 s, `sbatch` rejected | wrong/again `--account` | use `--account=project_462001520` (set in `submit_regen.sh`) |
| mechanism in results ≠ what you set | LLM overrode it | run with `ADMET_USE_LLM=0` |
| `n=25` arms identical / high variance with `mt_cotrain` | few-shot fallback ignores aux | use `pretrain_finetune` (see RESULTS.md) |

Notes:
- The login node's system `python3` is too old to import the stack — **all** Python must run in the
  container venv (use `csh`).
- A `both`-arm `pretrain_finetune` job is the slowest (~8 min: pretrains on thousands of aux rows);
  `external` is faster, `baseline` (GBM) ~seconds. Size `--time` accordingly (default 6 h is ample).

---

## 10. Quick recipes

```bash
# A) New mechanism on the 3 weak endpoints, few-shot, no LLM:
ADMET_USE_LLM=0 bash $WORK/further-improvements/scripts/submit_regen.sh \
  $WORK/further-improvements/conf/fewshot-pretrain_finetune.yaml \
  $WORK/further-improvements/results-fewshot/pf

# B) Smoke one cell first, then gate the full sweep on it:
SMOKE=$(ADMET_USE_LLM=0 bash .../submit_regen.sh .../conf/smoke-fewshot.yaml .../results-fewshot/smoke | awk '/submitted/{print $NF}')
SBATCH_DEP=$SMOKE ADMET_USE_LLM=0 bash .../submit_regen.sh .../conf/fewshot-pretrain_finetune.yaml .../results-fewshot/pf

# C) Reproduce an original cell WITH preds (to enable molecule-level stats):
#    edit a conf to mechanism mt_cotrain, submit, then compare_fewshot / bootstrap_lift_molecule.
```
