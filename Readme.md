
# Team 3 — ADMET few-shot transfer learning

Implementation of the multi-task few-shot ADMET pipeline described in [Plan.md](Plan.md).
Designed to run on Lumi-G via SLURM array jobs inside a Singularity container.

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

## Development

```
pip install -e '.[dev]'           # plus '[chem]' for rdkit/chemprop, '[external]' for Polaris/TDC, '[llm]' for Aitta
PYTHONPATH=src python -m pytest tests/
```

