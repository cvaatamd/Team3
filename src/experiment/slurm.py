"""SLURM dispatcher (Lumi-G shape per Readme.md).

Strategy:

* One sbatch array script per *sweep*: `--array=0-N%C` runs each (endpoint, arm, n, seed) job
  as a separate array task and caps concurrency at `C`.
* Each array task reads `$SLURM_ARRAY_TASK_ID` and runs `admet-run-job` with that index
  against the JSONL manifest. Worker is launched inside `singularity exec` using the
  chemprop venv on disk, matching the Pytorch-container recipe in Readme.md.
* For clarity we emit *two* files: `<job>.sbatch` and `<job>.worker.sh`. The sbatch script
  only orchestrates SLURM; the worker contains the container/venv plumbing. Easier to debug
  and easier to run by hand on a single interactive node.

A second mode, `submit_sweep_local`, replays the array locally with `subprocess` — useful for
sanity-checking the wiring on a laptop / single GPU node interactively.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Optional

import yaml

log = logging.getLogger(__name__)


@dataclass
class SlurmConfig:
    account: str
    partition: str = "small-g"
    time: str = "12:00:00"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 8
    gpus_per_node: int = 1
    mem: str = "32G"

    container_sif: str = "/appl/local/laifs/containers/lumi-multitorch-latest.sif"
    venv_activate: str = "chemprop/bin/activate"
    singularity_modules: list[str] = field(default_factory=lambda: ["lumi-aif-singularity-bindings"])
    module_use: list[str] = field(default_factory=lambda: ["/appl/local/laifs/modules"])
    binds: list[str] = field(default_factory=list)

    work_dir: str = "${HOME}/admet-fewshot"
    results_dir: str = "${HOME}/admet-fewshot/results"
    logs_dir: str = "${HOME}/admet-fewshot/results/logs"

    array_concurrency: int = 16

    @classmethod
    def from_yaml(cls, path: Path) -> "SlurmConfig":
        raw = yaml.safe_load(path.read_text())
        return cls(**raw)


# --- Templates --------------------------------------------------------------
# Python's string.Template uses `$name`; `$$` becomes a literal `$` for shell.

SBATCH_TEMPLATE = Template(r"""#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --time=$time
#SBATCH --nodes=$nodes
#SBATCH --ntasks=$ntasks
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --gpus-per-node=$gpus_per_node
#SBATCH --mem=$mem
#SBATCH --array=0-$array_max%$array_concurrency
#SBATCH --output=$logs_dir/$job_name-%A_%a.out
#SBATCH --error=$logs_dir/$job_name-%A_%a.err

set -euo pipefail
mkdir -p "$results_dir" "$logs_dir"

echo "[slurm] host=$$(hostname) job=$$SLURM_JOB_ID task=$$SLURM_ARRAY_TASK_ID"

srun bash "$worker_path" "$$SLURM_ARRAY_TASK_ID"
""")


WORKER_TEMPLATE = Template(r"""#!/bin/bash
# Per-task worker: prepare environment, then run one job from the manifest.
# Argument: array task index (== JSONL line number).

set -euo pipefail
TASK_IDX="$${1:?usage: $$0 ARRAY_TASK_INDEX}"

WORK_DIR="$work_dir"
RESULTS_DIR="$results_dir"
MANIFEST="$manifest_path"
CONTAINER="$container_sif"
VENV_ACTIVATE="$venv_activate"

# LLM plumbing — only used if ADMET_USE_LLM=1 is set. Provider is decided by conf/llm.yaml,
# so we forward both possible API keys; whichever the active provider needs is read at runtime.
export ADMET_USE_LLM="$${ADMET_USE_LLM:-$use_llm_flag}"
# Tokens must be inherited from the submitting environment (sbatch passes env by default;
# if not, set --export=ALL). Workers fail loudly at the first LLM call if the configured
# provider's key isn't set.
export AITTA_API_TOKEN="$${AITTA_API_TOKEN:-}"
export ANTHROPIC_API_KEY="$${ANTHROPIC_API_KEY:-}"
# Forward into the singularity container.
export SINGULARITYENV_ADMET_USE_LLM="$$ADMET_USE_LLM"
export SINGULARITYENV_AITTA_API_TOKEN="$$AITTA_API_TOKEN"
export SINGULARITYENV_ANTHROPIC_API_KEY="$$ANTHROPIC_API_KEY"

# Lumi container modules.
module purge
$module_use_lines
$module_load_lines

cd "$$WORK_DIR"

singularity exec \
$bind_args    --pwd "$$WORK_DIR" \
    "$$CONTAINER" \
    bash -lc "source '$$VENV_ACTIVATE' && \
              python -m experiment.cli run-job \
                --manifest '$$MANIFEST' \
                --index '$$TASK_IDX' \
                --results-dir '$$RESULTS_DIR'"
""")


def render_sbatch(
    *,
    cfg: SlurmConfig,
    manifest_path: Path,
    worker_path: Path,
    array_max: int,
    job_name: str = "admet-sweep",
) -> str:
    return SBATCH_TEMPLATE.substitute(
        job_name=job_name,
        account=cfg.account,
        partition=cfg.partition,
        time=cfg.time,
        nodes=cfg.nodes,
        ntasks=cfg.ntasks,
        cpus_per_task=cfg.cpus_per_task,
        gpus_per_node=cfg.gpus_per_node,
        mem=cfg.mem,
        array_max=array_max,
        array_concurrency=cfg.array_concurrency,
        logs_dir=cfg.logs_dir,
        results_dir=cfg.results_dir,
        worker_path=str(worker_path),
    )


def render_worker(*, cfg: SlurmConfig, manifest_path: Path, use_llm: bool = False) -> str:
    module_use_lines = "\n".join(f"module use {p}" for p in cfg.module_use) or "true"
    module_load_lines = "\n".join(f"module load {m}" for m in cfg.singularity_modules) or "true"
    bind_args = "".join(f"    --bind {b} \\\n" for b in cfg.binds)
    return WORKER_TEMPLATE.substitute(
        work_dir=cfg.work_dir,
        results_dir=cfg.results_dir,
        manifest_path=str(manifest_path),
        container_sif=cfg.container_sif,
        venv_activate=cfg.venv_activate,
        module_use_lines=module_use_lines,
        module_load_lines=module_load_lines,
        bind_args=bind_args,
        use_llm_flag="1" if use_llm else "0",
    )


def submit_sweep(
    *,
    cfg: SlurmConfig,
    manifest_path: Path,
    n_jobs: int,
    job_name: str = "admet-sweep",
    dry_run: bool = False,
    use_llm: bool = False,
) -> tuple[Path, Path, Optional[str]]:
    """Write sbatch + worker scripts next to the manifest and (optionally) submit.

    Returns (sbatch_path, worker_path, slurm_job_id_or_None).
    """
    if n_jobs <= 0:
        raise ValueError("manifest is empty")
    worker_path = manifest_path.with_name(f"{job_name}.worker.sh")
    sbatch_path = manifest_path.with_name(f"{job_name}.sbatch")

    worker_path.write_text(render_worker(cfg=cfg, manifest_path=manifest_path, use_llm=use_llm))
    worker_path.chmod(worker_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    sbatch_path.write_text(render_sbatch(
        cfg=cfg, manifest_path=manifest_path, worker_path=worker_path,
        array_max=n_jobs - 1, job_name=job_name,
    ))
    sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    log.info("wrote sbatch script -> %s", sbatch_path)
    log.info("wrote worker script -> %s", worker_path)

    if dry_run:
        log.info("dry-run: not calling sbatch")
        return sbatch_path, worker_path, None
    if shutil.which("sbatch") is None:
        log.warning("sbatch not on PATH — leaving scripts for manual submission")
        return sbatch_path, worker_path, None
    res = subprocess.run(
        ["sbatch", "--parsable", str(sbatch_path)],
        check=True, capture_output=True, text=True,
    )
    job_id = res.stdout.strip()
    log.info("submitted SLURM job %s", job_id)
    return sbatch_path, worker_path, job_id


def submit_sweep_local(
    *,
    manifest_path: Path,
    n_jobs: int,
    results_dir: Path,
    python_executable: str | None = None,
    env: dict | None = None,
) -> list[int]:
    """Run the array locally by iterating indices and calling `admet-run-job`."""
    py = python_executable or shutil.which("python") or "python"
    env = env if env is not None else os.environ.copy()
    rcs = []
    for i in range(n_jobs):
        cmd = [
            py, "-m", "experiment.cli", "run-job",
            "--manifest", str(manifest_path),
            "--index", str(i),
            "--results-dir", str(results_dir),
        ]
        log.info("[local %d/%d] %s", i + 1, n_jobs, " ".join(cmd))
        res = subprocess.run(cmd, check=False, env=env)
        rcs.append(res.returncode)
    return rcs
