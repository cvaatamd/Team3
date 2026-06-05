"""Top-level CLI: enumerate, submit, run-job, collect.

Three subcommands:

* `admet-submit-sweep` — load plan YAML, enumerate jobs to a JSONL manifest, render+submit
  the SLURM array, and (optionally) wait. With `--dry-run` it stops after writing the script
  so you can review it. With `--local` it runs the array in-process instead of via sbatch.

* `admet-run-job` — the per-task worker. Reads one line of the manifest by index and runs it.
  This is what each SLURM array task invokes.

* `admet-plan` — collect per-job result JSONs into the aggregate table, plots, and report.

Why this shape: it separates *enumeration / dispatch* from *one-shot execution*. The dispatch
side knows about SLURM, accounts, time limits. The execution side knows nothing about SLURM —
it just reads a manifest index. That makes the worker reusable from a laptop, a local node, or
any non-SLURM scheduler.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .runner import collect_results, load_plan, read_manifest, run_one, write_manifest
from .slurm import SlurmConfig, submit_sweep, submit_sweep_local

log = logging.getLogger(__name__)

DEFAULT_PLAN = Path("conf/sweep.yaml")
DEFAULT_SLURM = Path("conf/slurm.yaml")
DEFAULT_LLM = Path("conf/llm.yaml")


def _logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


# -- submit-sweep -----------------------------------------------------------

def submit_sweep_cli(argv: list[str] | None = None) -> int:
    _logging()
    p = argparse.ArgumentParser("admet-submit-sweep")
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="sweep plan YAML")
    p.add_argument("--slurm", type=Path, default=DEFAULT_SLURM, help="slurm config YAML")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--job-name", default="admet-sweep")
    p.add_argument("--dry-run", action="store_true",
                   help="write sbatch+worker but don't call sbatch")
    p.add_argument("--local", action="store_true",
                   help="run the array locally with subprocess (no SLURM)")
    p.add_argument("--llm", action="store_true",
                   help="use Aitta-backed LLM agents in workers (sets ADMET_USE_LLM=1)")
    args = p.parse_args(argv)

    plan = load_plan(args.plan)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.results_dir / "manifest.jsonl"
    write_manifest(plan, manifest)
    jobs = read_manifest(manifest)

    if args.local:
        import os
        env = os.environ.copy()
        if args.llm:
            env["ADMET_USE_LLM"] = "1"
        rcs = submit_sweep_local(
            manifest_path=manifest, n_jobs=len(jobs), results_dir=args.results_dir, env=env,
        )
        n_fail = sum(1 for r in rcs if r != 0)
        log.info("local sweep complete: %d/%d ok", len(rcs) - n_fail, len(rcs))
        return 0 if n_fail == 0 else 1

    cfg = SlurmConfig.from_yaml(args.slurm)
    sbatch, worker, job_id = submit_sweep(
        cfg=cfg, manifest_path=manifest, n_jobs=len(jobs),
        job_name=args.job_name, dry_run=args.dry_run, use_llm=args.llm,
    )
    if job_id:
        print(job_id)
    else:
        print(f"sbatch: {sbatch}")
        print(f"worker: {worker}")
        print(f"manifest: {manifest}  ({len(jobs)} jobs)")
    return 0


# -- run-job ----------------------------------------------------------------

def run_job_cli(argv: list[str] | None = None) -> int:
    _logging()
    p = argparse.ArgumentParser("admet-run-job")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--index", type=int, required=True,
                   help="manifest line index (== $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--force", action="store_true", help="re-run even if result exists")
    p.add_argument("--llm", action="store_true",
                   help="use Aitta-backed LLM agents (overrides $ADMET_USE_LLM)")
    p.add_argument("--llm-config", type=Path, default=DEFAULT_LLM)
    args = p.parse_args(argv)
    run_one(
        manifest_path=args.manifest,
        index=args.index,
        results_dir=args.results_dir,
        skip_if_exists=not args.force,
        use_llm=True if args.llm else None,
        llm_config_path=args.llm_config if args.llm_config.exists() else None,
    )
    return 0


# -- plan / collect ---------------------------------------------------------

def plan_cli(argv: list[str] | None = None) -> int:
    _logging()
    p = argparse.ArgumentParser("admet-plan")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--llm", action="store_true",
                   help="use Aitta-backed LLMPlanner to write the §0 characterization")
    p.add_argument("--llm-config", type=Path, default=DEFAULT_LLM)
    args = p.parse_args(argv)
    df = collect_results(args.results_dir)
    if df.empty:
        log.warning("no results found in %s/runs", args.results_dir)
        return 1
    out = args.results_dir / "results.parquet"
    df.to_parquet(out)
    log.info("collected %d results -> %s", len(df), out)

    # build curves + report via the planner's helpers
    from agents.planner import Planner
    from data.expansionrx import load_expansionrx
    plan = load_plan(args.plan)
    data = load_expansionrx()
    if args.llm:
        from agents.llm import LLMConfig, build_client
        from agents.llm_overrides import LLMPlanner
        cfg = LLMConfig.from_yaml(args.llm_config) if args.llm_config.exists() \
            else LLMConfig()
        planner = LLMPlanner(plan=plan, results_dir=args.results_dir, data=data,
                             llm=build_client(cfg))
    else:
        planner = Planner(plan=plan, results_dir=args.results_dir, data=data)
    from eval.curves import plot_curves, plot_ma_rae
    plot_curves(df, args.results_dir / "curves.png")
    plot_ma_rae(df, args.results_dir / "ma_rae.png")
    (args.results_dir / "report.md").write_text(planner._write_report(df))
    log.info("wrote curves + report.md to %s", args.results_dir)
    return 0


# -- entry points -----------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Dispatcher: `python -m experiment.cli <submit-sweep|run-job|collect> ...`."""
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python -m experiment.cli <submit-sweep|run-job|collect> [...]",
              file=sys.stderr)
        return 2
    sub, rest = argv[0], argv[1:]
    if sub in ("submit-sweep", "submit_sweep"):
        return submit_sweep_cli(rest)
    if sub in ("run-job", "run_job"):
        return run_job_cli(rest)
    if sub in ("collect", "plan"):
        return plan_cli(rest)
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
