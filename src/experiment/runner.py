"""Job enumeration + single-job worker.

Two responsibilities:

1. Build an `ExperimentPlan` from a YAML, enumerate it into JobSpecs, and write them to a
   JSONL manifest. The SLURM array index maps 1:1 to the JSONL line index.

2. Run *one* job end-to-end given a manifest path + index. This is the entry point that each
   SLURM array task invokes. It also works fine in-process for laptop dev.

Results are written to `results_dir/runs/<job_id>.json`. The planner collects them into the
aggregate table when the array job finishes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from agents.contracts import EvalResult, ExperimentPlan, JobSpec, PoolRequest
from agents.data_agent import DataAgent
from agents.ml_agent import MLAgent, TrainSpec
from agents.planner import _pool_request_from_job
from data.expansionrx import ExpansionRxData, load_expansionrx

log = logging.getLogger(__name__)


# ---- Plan loading + enumeration -------------------------------------------

def load_plan(path: Path) -> ExperimentPlan:
    raw = yaml.safe_load(path.read_text())
    # YAML loads `null` -> Python None; keep `n_grid` as Optional[int] entries.
    return ExperimentPlan(
        target_endpoints=raw["target_endpoints"],
        arms=raw["arms"],
        n_grid=raw["n_grid"],
        seeds=raw["seeds"],
        mechanism_by_arm=raw["mechanism_by_arm"],
        exclude_flagged_slices=raw.get("exclude_flagged_slices", True),
        rae_definition=raw.get("rae_definition", "range_normalized"),
    )


def write_manifest(plan: ExperimentPlan, out_path: Path) -> Path:
    jobs = plan.enumerate_jobs()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for j in jobs:
            f.write(j.model_dump_json() + "\n")
    log.info("wrote %d jobs to %s", len(jobs), out_path)
    return out_path


def read_manifest(path: Path) -> list[JobSpec]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(JobSpec.model_validate_json(line))
    return out


_ARM_ORDER = ["baseline", "intra_task", "external", "both"]


def plan_from_manifest(path: Path) -> ExperimentPlan:
    """Reconstruct the ExperimentPlan from a results dir's manifest.

    This keeps `collect` self-describing: the report header reflects exactly what was run for
    *this* results dir, with no dependency on which plan YAML happened to be the default. Useful
    for single-endpoint / custom sweeps where the default plan would mislabel the header.
    """
    jobs = read_manifest(path)
    if not jobs:
        raise ValueError(f"empty manifest: {path}")

    endpoints, arms, ns, seeds = [], [], [], []
    mech_by_arm: dict[str, str] = {}
    for j in jobs:
        if j.target_endpoint not in endpoints:
            endpoints.append(j.target_endpoint)
        if j.arm not in arms:
            arms.append(j.arm)
        if j.n not in ns:
            ns.append(j.n)
        if j.seed not in seeds:
            seeds.append(j.seed)
        mech_by_arm.setdefault(j.arm, j.mechanism)

    arms = sorted(arms, key=lambda a: (_ARM_ORDER.index(a) if a in _ARM_ORDER else len(_ARM_ORDER)))
    n_grid = sorted([n for n in ns if n is not None]) + ([None] if None in ns else [])
    seeds = sorted(seeds)
    return ExperimentPlan(
        target_endpoints=endpoints,
        arms=arms,
        n_grid=n_grid,
        seeds=seeds,
        mechanism_by_arm=mech_by_arm,
        exclude_flagged_slices=jobs[0].exclude_flagged_slices,
        rae_definition=jobs[0].rae_definition,
    )


# ---- Single-job worker -----------------------------------------------------

@dataclass
class RunEnv:
    manifest_path: Path
    runs_dir: Path
    log: dict


def _build_agents(data: ExpansionRxData, use_llm: bool, llm_config_path: Path | None = None):
    """Construct (DataAgent, MLAgent). LLM-driven variants if `use_llm` (Aitta-backed)."""
    if not use_llm:
        return DataAgent(data), MLAgent()
    # Lazy import — the LLM module pulls in openai, which is an optional extra.
    from agents.llm import AittaClient, AittaConfig
    from agents.llm_overrides import LLMDataAgent, LLMMLAgent
    cfg = AittaConfig.from_yaml(llm_config_path) if llm_config_path else AittaConfig()
    client = AittaClient(cfg)
    return LLMDataAgent(data, client), LLMMLAgent(client)


def run_one(
    manifest_path: Path,
    index: int,
    results_dir: Path,
    *,
    cache_dataset_dir: Optional[Path] = None,
    skip_if_exists: bool = True,
    use_llm: bool | None = None,
    llm_config_path: Path | None = None,
) -> EvalResult:
    jobs = read_manifest(manifest_path)
    if not (0 <= index < len(jobs)):
        raise IndexError(f"index {index} out of range for manifest of size {len(jobs)}")
    job = jobs[index]

    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{job.job_id}.json"
    if skip_if_exists and out_path.exists():
        log.info("[%s] already done, skipping", job.job_id)
        return EvalResult.model_validate_json(out_path.read_text())

    log.info("[%s] starting on host=%s pid=%s", job.job_id, socket.gethostname(), os.getpid())
    t0 = time.time()
    data = load_expansionrx()  # cached on HF disk; OK to re-load per job
    if use_llm is None:
        use_llm = os.environ.get("ADMET_USE_LLM", "0") == "1"
    da, ma = _build_agents(data, use_llm=use_llm, llm_config_path=llm_config_path)

    pool = da.build_pool(_pool_request_from_job(job))
    spec = TrainSpec(
        pool=pool, test_df=data.test, arm=job.arm, n=job.n, seed=job.seed,
        mechanism=job.mechanism, rae_definition=job.rae_definition,
    )
    result = ma.run(spec)
    # stamp environment provenance
    result.provenance.update({
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "elapsed_s": time.time() - t0,
        "manifest_hash": _hash_file(manifest_path),
    })
    out_path.write_text(result.model_dump_json(indent=2))
    log.info("[%s] done in %.1fs, RAE=%.4f", job.job_id, time.time() - t0, result.rae)
    return result


def collect_results(results_dir: Path) -> pd.DataFrame:
    """Gather all per-job JSONs into a tidy DataFrame."""
    runs_dir = results_dir / "runs"
    rows = []
    if not runs_dir.exists():
        return pd.DataFrame()
    for p in sorted(runs_dir.glob("*.json")):
        try:
            r = EvalResult.model_validate_json(p.read_text())
        except Exception as e:
            log.warning("could not parse %s: %s", p, e)
            continue
        row = r.model_dump()
        # flatten extra_metrics
        em = row.pop("extra_metrics", {}) or {}
        row.update(em)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.rename(columns={"target_endpoint": "endpoint"})
    return df


# ---- helpers ---------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
