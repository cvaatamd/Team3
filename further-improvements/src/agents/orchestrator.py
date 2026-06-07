"""Standalone ADMET agent — the full workflow in one Python process, driven by Aitta.

This is the single entry point that ties the three agents together end-to-end, with no
dependency on Cursor or any external IDE/agent framework. It is the Python embodiment of the
workflow diagram:

    Planner ──► Data agent ──► ML agent ──► Planner (interpretation)
        │  (enumerate sweep)  (sources)   (mechanism)   (characterization report)

All three agentic decision points are routed through CSC's Aitta inference service
(https://aitta.csc.fi) using the OpenAI-compatible API:

  * Data agent  — `LLMDataAgent.select_sources`  (which external assays to fold in)
  * ML agent    — `LLMMLAgent.choose_mechanism`  (mt_cotrain / pretrain_finetune / ...)
  * Planner     — `LLMPlanner._write_report`      (the written characterization)

Two execution modes:

  --mode slurm  (default)  Dispatch the sweep as a SLURM array on LUMI-G and chain a dependent
                           collect job. Run this from a LUMI **login node**. Scales to the full
                           240-job sweep; you can log out while SLURM runs it.

  --mode local             Run the entire sweep *in this process* (plan → data → train → eval →
                           interpret). Run this **inside the container on a compute node**
                           (e.g. via `sbatch scripts/agent.sbatch`). This is the cleanest
                           single-process demonstration of the agent; use a small plan
                           (conf/sweep-mini.yaml) so it finishes on one GPU.

Reproducible by hand, independent of Cursor:

    # one-shot, parallel, scalable (login node):
    export AITTA_API_TOKEN=...                 # from https://aitta-auth.csc.fi/myToken
    python -m agents.orchestrator --mode slurm --plan conf/sweep.yaml

    # one process, single GPU, full agent loop (compute node, in container):
    python -m agents.orchestrator --mode local --plan conf/sweep-mini.yaml

    # check Aitta connectivity + the plan without launching anything:
    python -m agents.orchestrator --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("admet.agent")

DEFAULT_PLAN = Path("conf/sweep.yaml")
DEFAULT_SLURM = Path("conf/slurm.yaml")
DEFAULT_LLM = Path("conf/llm.yaml")


# --------------------------------------------------------------------------- #
#  The agent                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class ADMETAgent:
    plan_path: Path = DEFAULT_PLAN
    slurm_path: Path = DEFAULT_SLURM
    llm_config_path: Path = DEFAULT_LLM
    results_dir: Path = Path("results")
    job_name: str = "admet-sweep"
    use_llm: bool = True
    wait: bool = False

    # ---- Aitta connectivity ------------------------------------------------ #

    def _aitta_config(self):
        from agents.llm import AittaConfig
        if self.llm_config_path and self.llm_config_path.exists():
            return AittaConfig.from_yaml(self.llm_config_path)
        return AittaConfig()

    def aitta_preflight(self, *, warm_up: bool = True) -> bool:
        """Check Aitta is reachable and (optionally) bring the model online before we commit.

        Uses the documented Aitta service endpoints (/status, /worker/<model>) and a tiny
        warm-up chat to trigger on-demand model allocation. Returns True if the model is
        ready (or we deliberately skipped LLM use).
        """
        if not self.use_llm:
            log.info("LLM disabled (--no-llm): skipping Aitta preflight")
            return True
        from agents.llm import AittaClient
        cfg = self._aitta_config()
        client = AittaClient(cfg)

        status = client.service_status()
        if "error" in status:
            log.warning("Aitta /status check failed: %s", status["error"])
        elif status.get("status", "OK") != "OK" or "reason" in status:
            log.warning("Aitta reports a maintenance window: %s", status)
        else:
            log.info("Aitta service status: OK")

        log.info("Aitta model: %s (base_url=%s)", cfg.model, cfg.base_url)
        workers = client.online_workers(cfg.model)
        log.info("online workers for %s: %s", cfg.model, workers or "(none)")

        if not warm_up:
            return client.is_model_online(cfg.model)
        ok = client.ensure_online(cfg.model)
        if not ok:
            log.warning(
                "Aitta model %s is not online yet. The sweep can still proceed; workers will "
                "retry/back off, but expect a multi-minute cold-start on the first call.",
                cfg.model,
            )
        return ok

    # ---- Planning ---------------------------------------------------------- #

    def build_manifest(self) -> tuple[Path, int]:
        """Planner step: enumerate the sweep into a JSONL manifest. Returns (path, n_jobs)."""
        from experiment.runner import load_plan, read_manifest, write_manifest
        plan = load_plan(self.plan_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        manifest = (self.results_dir / "manifest.jsonl").resolve()
        write_manifest(plan, manifest)
        n_jobs = len(read_manifest(manifest))
        log.info(
            "plan: %d jobs over endpoints=%s arms=%s n_grid=%s seeds=%s",
            n_jobs, plan.target_endpoints, plan.arms, plan.n_grid, plan.seeds,
        )
        return manifest, n_jobs

    # ---- Execution: SLURM -------------------------------------------------- #

    def run_slurm(self, *, dry_run: bool = False) -> int:
        from experiment.slurm import SlurmConfig, submit_sweep

        manifest, n_jobs = self.build_manifest()
        cfg = SlurmConfig.from_yaml(self.slurm_path)
        results_abs = self.results_dir.resolve()
        cfg.results_dir = str(results_abs)
        cfg.logs_dir = str(results_abs / "logs")
        cfg.work_dir = str(Path.cwd().resolve())

        sbatch, worker, job_id = submit_sweep(
            cfg=cfg, manifest_path=manifest, n_jobs=n_jobs,
            job_name=self.job_name, dry_run=dry_run, use_llm=self.use_llm,
        )
        if dry_run or job_id is None:
            log.info("sweep prepared but not submitted (dry_run=%s).", dry_run)
            log.info("  sbatch:   %s", sbatch)
            log.info("  worker:   %s", worker)
            log.info("  manifest: %s (%d jobs)", manifest, n_jobs)
            return 0

        log.info("submitted sweep array as SLURM job %s", job_id)
        collect_id = self._submit_collect(job_id, results_abs)
        if collect_id:
            log.info("submitted dependent collect as SLURM job %s "
                     "(runs after the sweep finishes)", collect_id)

        print(f"sweep_job_id={job_id}")
        if collect_id:
            print(f"collect_job_id={collect_id}")
        print(f"results_dir={results_abs}")
        log.info("Done. Check progress with `squeue -u $USER`; read %s/report.md when finished.",
                 results_abs)

        if self.wait:
            ids = [i for i in (job_id, collect_id) if i]
            self._wait_for(ids)
        return 0

    def _submit_collect(self, sweep_job_id: str, results_abs: Path) -> Optional[str]:
        """Chain the collect+interpret step to run after the sweep (afterany)."""
        collect_script = Path("scripts/collect_llm.sbatch").resolve()
        if not collect_script.exists():
            log.warning("collect script %s missing; collect not chained. "
                        "Run `python -m experiment.cli collect --llm --results-dir %s` "
                        "manually when the sweep finishes.", collect_script, results_abs)
            return None
        if _which("sbatch") is None:
            log.warning("sbatch not on PATH; cannot chain collect.")
            return None
        logs = results_abs / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["RESULTS_DIR"] = str(results_abs)
        cmd = [
            "sbatch", "--parsable",
            f"--dependency=afterany:{sweep_job_id}",
            "--output", str(logs / "collect-%j.out"),
            "--error", str(logs / "collect-%j.err"),
            str(collect_script),
        ]
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            log.warning("could not submit collect job: %s", e.stderr.strip())
            return None

    @staticmethod
    def _wait_for(job_ids: list[str], *, poll_s: int = 60) -> None:
        if _which("squeue") is None:
            log.info("squeue not available; not waiting.")
            return
        pending = set(job_ids)
        log.info("waiting for SLURM jobs %s (poll every %ds)...", sorted(pending), poll_s)
        while pending:
            time.sleep(poll_s)
            res = subprocess.run(
                ["squeue", "-h", "-j", ",".join(pending), "-o", "%i"],
                capture_output=True, text=True,
            )
            still = {ln.split("_")[0].strip() for ln in res.stdout.splitlines() if ln.strip()}
            done = {j for j in pending if not any(j == s for s in still)}
            if done:
                log.info("finished: %s", sorted(done))
            pending = {j for j in pending if j in still}
        log.info("all SLURM jobs finished.")

    # ---- Execution: local (single process) --------------------------------- #

    def run_local(self) -> int:
        """Run the whole agent loop in this process: plan → data → train → eval → interpret.

        Must run inside the chemprop container on a GPU compute node. With LLM enabled, the
        Data/ML/Planner decisions are routed through Aitta.
        """
        from agents.planner import Planner, in_process_dispatcher
        from data.expansionrx import load_expansionrx
        from experiment.runner import load_plan

        plan = load_plan(self.plan_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        log.info("loading ExpansionRx dataset...")
        data = load_expansionrx()

        if self.use_llm:
            from agents.llm import AittaClient
            from agents.llm_overrides import LLMDataAgent, LLMMLAgent, LLMPlanner
            client = AittaClient(self._aitta_config())
            data_agent = LLMDataAgent(data, client)
            ml_agent = LLMMLAgent(client)
            planner = LLMPlanner(plan=plan, results_dir=self.results_dir, data=data, llm=client)
            log.info("agents: LLM-driven (Aitta) Data/ML/Planner")
        else:
            from agents.data_agent import DataAgent
            from agents.ml_agent import MLAgent
            data_agent = DataAgent(data)
            ml_agent = MLAgent()
            planner = Planner(plan=plan, results_dir=self.results_dir, data=data)
            log.info("agents: deterministic Data/ML/Planner (no LLM)")

        dispatch = in_process_dispatcher(
            data, data.test, data_agent=data_agent, ml_agent=ml_agent,
        )
        n_jobs = len(planner.jobs())
        log.info("running %d jobs in-process (this can take a while on a single GPU)...", n_jobs)
        report_path = planner.run(dispatch)
        log.info("done. report: %s", report_path)
        print(f"report={report_path}")
        print(f"results_dir={self.results_dir.resolve()}")
        return 0


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #

def _which(name: str) -> Optional[str]:
    import shutil
    return shutil.which(name)


def _logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    _logging()
    p = argparse.ArgumentParser(
        "admet-agent",
        description="Standalone Aitta-driven ADMET few-shot transfer agent (runs on LUMI).",
    )
    p.add_argument("--mode", choices=["slurm", "local"], default="slurm",
                   help="slurm: dispatch a SLURM array from a login node (default). "
                        "local: run the whole sweep in this process (compute node).")
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="sweep plan YAML")
    p.add_argument("--slurm", type=Path, default=DEFAULT_SLURM, help="SLURM config YAML")
    p.add_argument("--llm-config", type=Path, default=DEFAULT_LLM, help="Aitta config YAML")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--job-name", default="admet-sweep")
    p.add_argument("--mini", action="store_true",
                   help="shortcut: use conf/sweep-mini.yaml and results-mini/")
    p.add_argument("--no-llm", action="store_true",
                   help="use deterministic agents instead of Aitta-backed ones")
    p.add_argument("--no-warmup", action="store_true",
                   help="check Aitta status/workers but don't trigger model allocation")
    p.add_argument("--wait", action="store_true",
                   help="(slurm mode) block until the sweep + collect jobs finish")
    p.add_argument("--dry-run", action="store_true",
                   help="run preflight + planning only; do not launch any jobs")
    args = p.parse_args(argv)

    if args.mini:
        args.plan = Path("conf/sweep-mini.yaml")
        if args.results_dir == Path("results"):
            args.results_dir = Path("results-mini")
        if args.job_name == "admet-sweep":
            args.job_name = "admet-mini"

    if not args.plan.exists():
        log.error("plan file not found: %s", args.plan)
        return 2

    agent = ADMETAgent(
        plan_path=args.plan,
        slurm_path=args.slurm,
        llm_config_path=args.llm_config,
        results_dir=args.results_dir,
        job_name=args.job_name,
        use_llm=not args.no_llm,
        wait=args.wait,
    )

    log.info("ADMET standalone agent | mode=%s plan=%s results=%s llm=%s",
             args.mode, args.plan, args.results_dir, agent.use_llm)

    online = agent.aitta_preflight(warm_up=not args.no_warmup)

    if args.dry_run:
        manifest, n_jobs = agent.build_manifest()
        log.info("dry-run complete: Aitta_ready=%s, %d jobs planned in %s",
                 online, n_jobs, manifest)
        return 0

    if args.mode == "slurm":
        return agent.run_slurm()
    return agent.run_local()


if __name__ == "__main__":
    sys.exit(main())
