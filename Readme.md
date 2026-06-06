
# Team 3 — ADMET few-shot transfer learning

Implementation of the multi-task few-shot ADMET pipeline described in [Plan.md](Plan.md).
Designed to run on Lumi-G via SLURM array jobs inside a Singularity container.

New here? Read [OVERVIEW.md](OVERVIEW.md) for the goal of the workflow and what you get when you
run it. Findings from the completed sweep are in [RESULTS.md](RESULTS.md). If Chemprop
`external`-arm jobs return `rae=null`, see [CHEMPROP_STABILITY.md](CHEMPROP_STABILITY.md).

## TL;DR — run the whole thing (no prior knowledge needed)

On a LUMI **login node**, from the project root:

```bash
# 1. One-time: get an Aitta token from https://aitta-auth.csc.fi/myToken, then:
echo 'PASTE_TOKEN_HERE' > conf/.aitta_token && chmod 600 conf/.aitta_token

# 2. Launch the full sweep + report (SLURM does the work; safe to log out after):
bash scripts/run_full_sweep.sh

# 3. Watch progress, then read the result:
bash scripts/check_status.sh
cat results/report.md
```

For a fast 8-job sanity check first: `bash scripts/run_full_sweep.sh --mini`.
Full guide + all options: `bash scripts/run_full_sweep.sh --help`.

| Script | What it does |
|---|---|
| `scripts/run_full_sweep.sh` | One command: checks prerequisites, submits the `conf/sweep.yaml` sweep + collect. `--mini`, `--yes`, `--dry-run`, `--token`, `--free-slot`. |
| `scripts/check_status.sh` | Read-only progress: queue, `N/240` runs done, which outputs are ready. |
| `scripts/submit_all.sh` | Lower-level: submit sweep array → dependent collect. Used by `run_full_sweep.sh`. |
| `scripts/submit_sweep.sh` | Render + submit only the sweep array. |
| `scripts/submit_collect.sh` | Submit only the collect/report step (when `runs/` already exists). |

Outputs land in `results/`: `results.parquet`, `curves.png`, `ma_rae.png`, `report.md`.

## Layout

```
conf/        sweep + endpoint + slurm YAML configs
src/data     ExpansionRx loader, transforms, external-source registry, harmonization
src/eval     RAE / MA-RAE metrics + learning-curve plotting
src/models   LightGBM baseline, Chemprop v2 MT wrapper, frozen-embedding head
src/agents   Typed contracts + deterministic data / ML / planner agents (§5 of Plan.md)
src/experiment   Job enumeration + SLURM dispatcher + worker CLI
tests/       Unit tests (metrics, transforms, contracts, sbatch rendering)
scripts/     submit_sweep.sh convenience wrapper
CHEMPROP_STABILITY.md   Chemprop + external-data NaN diagnosis and fixes (required reading for sparse arms)
```

## Quickstart — running the sweep

1. **Configure.** Edit [conf/slurm.yaml](conf/slurm.yaml) with your project account and any
   bind mounts your sweep needs. Edit [conf/sweep.yaml](conf/sweep.yaml) to pick endpoints,
   arms, the n-grid, and seeds.
2. **Submit the array.** From the project root on the login node:
   ```
   bash scripts/submit_sweep.sh                 # submits via sbatch
   bash scripts/submit_sweep.sh --dry-run       # render scripts only
   bash scripts/submit_sweep.sh --local         # run jobs in-process (no SLURM)
   ```
   This writes a `results/manifest.jsonl` of (endpoint, arm, n, seed) jobs, renders
   `admet-sweep.sbatch` + `admet-sweep.worker.sh`, and calls `sbatch`.
3. **Per-task execution.** Each SLURM array task runs the worker script, which loads the
   container + venv (see "Pytorch container" below) and invokes:
   ```
   python -m experiment.cli run-job --manifest results/manifest.jsonl \
       --index $SLURM_ARRAY_TASK_ID --results-dir results
   ```
   Results land in `results/runs/<job_id>.json`. Re-running an already-completed job is a
   no-op unless `--force` is passed.
4. **Collect.** After the array finishes:
   ```
   python -m experiment.cli collect --results-dir results
   ```
   Builds `results.parquet`, `curves.png`, `ma_rae.png`, and `report.md` (the
   characterization deliverable from §0 of Plan.md).

## Pytorch container

This will add chemprop to a venv that will be located on disk in the working directory, not the Pytorch container itself.

Based on https://docs.lumi-supercomputer.eu/laif/software/ai-environment/#add-more-pip-packages-to-container . 

### Adding chemprop to the Pytorch container 

```
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
singularity shell $SIF
Singularity> python -m venv chemprop --system-site-packages
Singularity> source chemprop/bin/activate
(chemprop) Singularity> pip install chemprop
```

### Using chemprop with the Pytorch container

```
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
run $SIF bash -c 'source chemprop/bin/activate && python -c "import chemprop; print(chemprop.__version__)"'
```

## ChemProp container

1. Prepare ChemProp container in Lumi
```shell
module load CrayEnv
module load cotainr
cotainr build chemprop.sif --system=lumi-g --conda-env=chemprop_env.yml
```

2. Running the container — use the project's SLURM dispatcher (`scripts/submit_sweep.sh`),
   which renders an sbatch array script following the pattern above and submits it. If you
   want to invoke the worker by hand on an interactive node:
   ```shell
   singularity exec --bind /projappl/project_XXXX:/projappl/project_XXXX \
                    --pwd /projappl/project_XXXX/wd \
                    container.sif \
                    bash -lc 'source chemprop/bin/activate && \
                              python -m experiment.cli run-job \
                                --manifest results/manifest.jsonl --index 0 \
                                --results-dir results'
   ```

## LLM agents (Aitta, CSC LUMI inference)

The three "agentic" decision points from Plan.md §5 can be routed to an LLM via
[Aitta](https://aitta.csc.fi/page/docs) (OpenAI-compatible, runs on Lumi):

| Decision | Class | Override point |
|---|---|---|
| Which external sources to include per target | `LLMDataAgent` | `select_sources` |
| Mechanism per pool shape (gbm / frozen / MT / pretrain-finetune) | `LLMMLAgent` | `choose_mechanism` |
| §0 characterization narrative | `LLMPlanner` | `_write_report` |

LLM calls live behind a small `AittaClient` (`src/agents/llm.py`) that wraps the `openai` SDK
against `https://aitta-api.csc.fi/openai/v1`, validates structured output against pydantic
schemas, retries on 429 with the documented ~60s cooldown, and **caches every decision to
disk** so re-runs don't burn tokens.

### Setup

1. Install the extra: `pip install -e '.[llm]'`
2. Get a token from <https://aitta-auth.csc.fi/myToken> (24h default; bind to your LUMI
   project for 90d) and export it: `export AITTA_API_TOKEN=...`
3. (Optional) tweak [`conf/llm.yaml`](conf/llm.yaml) — model, retries, cache dir.

### Usage

```
# Render + submit a SLURM sweep with LLM-driven workers
bash scripts/submit_sweep.sh --llm

# Run a single job locally with LLM agents
python -m experiment.cli run-job --manifest results/manifest.jsonl --index 0 --llm

# Write the characterization report with LLM narrative
python -m experiment.cli collect --llm
```

When `--llm` is passed to `submit-sweep`, the worker template exports `ADMET_USE_LLM=1` and
forwards `AITTA_API_TOKEN` into Singularity via `SINGULARITYENV_*`. The decision cache at
`results/llm_cache/` lives on the shared Lumi filesystem, so cache hits across worker tasks
are free even when many array jobs are running.

## Standalone agent (one Python process, Aitta-driven)

`src/agents/orchestrator.py` is a **single, self-contained agent** that runs the whole workflow
end to end with no dependency on Cursor or any external agent framework. It is the Python
embodiment of the Plan.md §5 loop:

```
Planner (enumerate sweep) -> Data agent (select_sources) -> ML agent (choose_mechanism)
                          -> train/eval -> Planner (write characterization report)
```

All three decision points are routed through Aitta (`openai/gpt-oss-120b` by default; see
`conf/llm.yaml`). On startup the agent does an **Aitta preflight**: it checks `/status`, lists
`/worker/<model>` to see if a worker is online, and (unless `--no-warmup`) sends a warm-up
request to trigger Aitta's on-demand model allocation before committing the sweep.

Two execution modes:

| Mode | Where to run | What it does |
|---|---|---|
| `--mode local` (the standalone agent) | GPU compute node, inside the container | Runs the full sweep **in one process**: plan → data → train → eval → interpret. Use a small plan (single GPU, serial). |
| `--mode slurm` | LUMI login node (needs both `sbatch` and the Python deps) | Dispatches the sweep as a SLURM array + a dependent collect job. Scales to the full 240-job sweep. |

### Run it (manually reproducible, no Cursor)

```bash
# Token once: https://aitta-auth.csc.fi/myToken
echo 'PASTE_TOKEN_HERE' > conf/.aitta_token && chmod 600 conf/.aitta_token

# Standalone agent as a single SLURM job (runs python -m agents.orchestrator --mode local):
bash scripts/run_agent.sh                # mini plan -> results-agent/
bash scripts/run_agent.sh --full         # full plan (serial on one GPU; slow)
bash scripts/run_agent.sh --no-llm       # deterministic agents (no Aitta)

# Aitta connectivity + plan check only (no jobs submitted):
bash scripts/run_agent.sh --check

# Parallel SLURM-array path instead (proven pipeline):
bash scripts/run_agent.sh --slurm
```

### Run the agent with the tasks on SLURM (parallel array)

`--mode local` runs every job serially in one process — fine for the mini plan, too slow for
the full 240-job sweep on a single GPU. To have the agent **dispatch the jobs as a parallel
SLURM array** (each `(endpoint, arm, n, seed)` is its own GPU task), use the SLURM path. From a
LUMI **login node** (where `sbatch` lives):

```bash
# Easiest: the launcher hands off to the proven submit pipeline
bash scripts/run_agent.sh --slurm            # full sweep -> results/

# Or drive the orchestrator's SLURM mode directly:
python -m agents.orchestrator --mode slurm --plan conf/sweep.yaml --results-dir results
```

What the SLURM path does:
1. **Plan** — enumerates the sweep into `results/manifest.jsonl`.
2. **Dispatch** — submits a SLURM **array** (`admet-sweep.sbatch`), one task per job, with
   concurrency capped in `conf/slurm.yaml` (`array_concurrency`, default 16).
3. **Collect (dependent job)** — chains a second SLURM job with
   `--dependency=afterany:<sweep_id>` that aggregates results, draws the plots, and writes the
   Aitta-narrated `report.md` after the array finishes.

It prints `sweep_job_id=...` and `collect_job_id=...`, then you can log out — SLURM runs it.
Monitor and read the result with:

```bash
squeue -u $USER                 # or: bash scripts/check_status.sh
cat results/report.md           # when the collect job is done
```

Add `--wait` to block until both the sweep and collect jobs finish:

```bash
python -m agents.orchestrator --mode slurm --plan conf/sweep.yaml --results-dir results --wait
```

> Note on environments: `--mode slurm` needs **both** `sbatch` (login node) and the Python deps
> (in the container). `scripts/run_agent.sh --slurm` handles this by delegating to
> `scripts/submit_all.sh`, which renders the array script inside the container and submits with
> `sbatch` on the login node. Use that wrapper unless you have a login-node Python env with the
> deps installed.

### Running a few rounds

Give each round its own results dir so they don't overwrite each other:

```bash
bash scripts/run_agent.sh --results-dir results-agent/round1
bash scripts/run_agent.sh --results-dir results-agent/round2
bash scripts/run_agent.sh --plan conf/sweep.yaml --results-dir results-agent/full1
```

- Re-running the **same** dir is a no-op for finished jobs (`<dir>/runs/*.json` are reused);
  use the `run-job --force` path to recompute.
- Aitta decisions are cached at `results/llm_cache` (a shared path set in `conf/llm.yaml`,
  keyed by inputs), so repeats across rounds are free and reproducible. Delete that folder, or
  point `decision_cache_dir` elsewhere, to force fresh LLM decisions.
- Tokens expire (24h default). If a round fails on auth, refresh
  `conf/.aitta_token` from <https://aitta-auth.csc.fi/myToken>.

You can also invoke the agent directly inside the container:

```bash
python -m agents.orchestrator --mode local --plan conf/sweep-mini.yaml --results-dir results-agent
python -m agents.orchestrator --dry-run                 # preflight + planning, launch nothing
```

| Script | What it does |
|---|---|
| `scripts/run_agent.sh` | Login-node launcher. Default submits `agent.sbatch`; `--check`, `--full`, `--no-llm`, `--slurm`. |
| `scripts/agent.sbatch` | Runs `python -m agents.orchestrator --mode local` in the container on a GPU node. Configurable via `PLAN`, `RESULTS_DIR`, `USE_LLM`. |

Outputs land in the chosen results dir: `results.parquet`, `curves.png`, `ma_rae.png`,
`report.md`.

## Full multi-endpoint experiment (5 rounds, parallel)

`scripts/run_fullexp.sh` runs a complete study: **5 endpoints**, each a full sweep (4 arms ×
n{25,50,100,250,500,full} × 5 seeds = 120 jobs), as parallel SLURM arrays with `array_concurrency:
100` (`conf/slurm-fullexp.yaml`). Output goes to a **separate** folder `results-fullexp/` and never
touches `results/` or `results-rounds/`. It is **disconnection-safe** — once submitted, SLURM runs
everything (sweeps + dependent collect jobs) without your session.

```bash
# token once: https://aitta-auth.csc.fi/myToken -> conf/.aitta_token
bash scripts/run_agent.sh --check                 # preflight (no jobs)
bash scripts/run_fullexp.sh --dry-run             # preview, submit nothing
nohup bash scripts/run_fullexp.sh &>results-fullexp/submit.log &   # launch, safe to log out
squeue -u $USER                                   # monitor
cat results-fullexp/<round>/report.md             # round in {hlm,mbpb,ksol,mppb,caco2eff}
```

Full step-by-step reproduction and the documented findings (with plots) live in
[results-fullexp/EXPERIMENT.md](results-fullexp/EXPERIMENT.md). Config: `conf/fullexp-*.yaml`,
`conf/slurm-fullexp.yaml`. The smaller 3-round exploration is in
[results-rounds/EXPERIMENT.md](results-rounds/EXPERIMENT.md) (`scripts/run_rounds.sh`).

### Metrics other than RAE

Every run stores `rae, mae, rmse, r2, spearman` in `results.parquet`, so other metrics need **no
re-training**. Re-aggregate plots + report for any metric (writes `curves_<metric>.png`,
`ma_<metric>.png`, `report_<metric>.md`; the default `rae` outputs are preserved):

```bash
python -m experiment.cli collect --results-dir results-fullexp/mppb --metric r2   # rae|mae|rmse|r2|spearman
# or on SLURM, with LLM narrative:  RESULTS_DIR=results-fullexp/mppb METRIC=r2 bash scripts/submit_collect.sh
```

The RAE *definition* itself is pluggable (`range_normalized` default, `vs_mean`, or the `official`
stub) via `rae_definition` in `src/eval/metrics.py`.

## Development

```
pip install -e '.[dev]'           # plus '[chem]' for rdkit/chemprop, '[external]' for Polaris/TDC, '[llm]' for Aitta
PYTHONPATH=src python -m pytest tests/
```

