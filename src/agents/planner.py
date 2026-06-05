"""Planner agent (§5.3).

Owns the sweep: enumerates (endpoint, arm, n, seed) jobs, dispatches to data + ML agents,
collects results, and writes the characterization report. Has a `dispatch_fn` hook so the same
class can run jobs in-process *or* hand them off to SLURM.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from data.expansionrx import ExpansionRxData
from eval.curves import plot_curves, plot_ma_rae
from .contracts import Arm, EvalResult, ExperimentPlan, JobSpec, PoolRequest
from .data_agent import DataAgent
from .ml_agent import MLAgent, TrainSpec

log = logging.getLogger(__name__)


DispatchFn = Callable[[list[JobSpec]], list[EvalResult]]


@dataclass
class Planner:
    plan: ExperimentPlan
    results_dir: Path
    data: ExpansionRxData

    def jobs(self) -> list[JobSpec]:
        return self.plan.enumerate_jobs()

    def run(self, dispatch_fn: DispatchFn) -> Path:
        """Dispatch all jobs, collect results, write the curves and report. Returns report path."""
        jobs = self.jobs()
        log.info("planner: %d jobs across %d endpoints, %d arms",
                 len(jobs), len(self.plan.target_endpoints), len(self.plan.arms))

        results = dispatch_fn(jobs)
        results_df = pd.DataFrame([r.model_dump() for r in results])
        if not results_df.empty:
            # unpack extra_metrics
            for col in ["mae", "rmse", "r2", "spearman", "n_test", "elapsed_s"]:
                results_df[col] = results_df["extra_metrics"].map(lambda d: d.get(col))
        results_path = self.results_dir / "results.parquet"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(results_path)

        # Curves
        if not results_df.empty:
            plot_curves(results_df, self.results_dir / "curves.png")
            plot_ma_rae(results_df, self.results_dir / "ma_rae.png")

        # Report
        report_path = self.results_dir / "report.md"
        report_path.write_text(self._write_report(results_df))
        log.info("planner: wrote results -> %s and report -> %s", results_path, report_path)
        return report_path

    def _write_report(self, df: pd.DataFrame) -> str:
        lines = [
            "# Multi-task few-shot ADMET — characterization",
            "",
            f"- Endpoints: {', '.join(self.plan.target_endpoints)}",
            f"- Arms: {', '.join(self.plan.arms)}",
            f"- n grid: {self.plan.n_grid}",
            f"- seeds: {self.plan.seeds}",
            f"- jobs run: {len(df)}",
            "",
        ]
        if df.empty:
            lines.append("_no results_")
            return "\n".join(lines)

        lines.append("## Best arm per (endpoint, n)\n")
        agg = (
            df.groupby(["target_endpoint", "arm", "n"], dropna=False)["rae"]
            .agg(["mean", "std"]).reset_index()
        )
        for ep, sub in agg.groupby("target_endpoint"):
            lines.append(f"### {ep}\n")
            piv = sub.pivot_table(index="n", columns="arm", values="mean")
            lines.append(piv.round(4).to_markdown())
            lines.append("")

        lines.append("\n## Interpretation\n")
        for ep, sub in agg.groupby("target_endpoint"):
            base = sub[sub["arm"] == "baseline"].set_index("n")["mean"]
            for arm in self.plan.arms:
                if arm == "baseline":
                    continue
                trial = sub[sub["arm"] == arm].set_index("n")["mean"]
                if trial.empty or base.empty:
                    continue
                deltas = (base - trial).dropna()
                if deltas.empty:
                    continue
                best_n = deltas.idxmax()
                lift = deltas.loc[best_n]
                lines.append(
                    f"- **{ep} / {arm}**: largest lift {lift:.4f} RAE at n={best_n} "
                    f"(baseline={base.loc[best_n]:.4f}, {arm}={trial.loc[best_n]:.4f})."
                )
        return "\n".join(lines)


def in_process_dispatcher(
    data: ExpansionRxData,
    test_df: pd.DataFrame,
    *,
    data_agent: Optional[DataAgent] = None,
    ml_agent: Optional[MLAgent] = None,
) -> DispatchFn:
    """A dispatcher that runs every job in this process (good for laptop dev / smoke tests)."""
    da = data_agent or DataAgent(data)
    ma = ml_agent or MLAgent()

    def _dispatch(jobs: list[JobSpec]) -> list[EvalResult]:
        out: list[EvalResult] = []
        for j in jobs:
            req = _pool_request_from_job(j)
            pool = da.build_pool(req)
            spec = TrainSpec(
                pool=pool, test_df=test_df, arm=j.arm, n=j.n, seed=j.seed,
                mechanism=j.mechanism, rae_definition=j.rae_definition,
            )
            out.append(ma.run(spec))
        return out

    return _dispatch


def _pool_request_from_job(j: JobSpec) -> PoolRequest:
    return PoolRequest(
        target_endpoint=j.target_endpoint,
        n=j.n,
        include_intra_task=j.arm in ("intra_task", "both"),
        include_external=j.arm in ("external", "both"),
        seed=j.seed,
        exclude_flagged_slices=j.exclude_flagged_slices,
    )
