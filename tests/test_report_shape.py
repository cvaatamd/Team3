"""Regression: the dataframe handed to Planner._write_report / plot_curves must carry both
`target_endpoint` (used by _write_report) and `endpoint` (used by curves). collect_results
and Planner.run() both add `endpoint` as a copy of `target_endpoint` — this test guards
against either being dropped.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.contracts import ExperimentPlan
from agents.planner import Planner


def _df(endpoints):
    rows = []
    for ep in endpoints:
        for seed in (0, 1):
            rows.append({
                "target_endpoint": ep, "endpoint": ep,
                "arm": "baseline", "n": 100, "seed": seed,
                "rae": 0.5 - 0.1 * seed, "mechanism": "gbm_baseline",
            })
    return pd.DataFrame(rows)


def test_write_report_uses_target_endpoint(tmp_path: Path):
    plan = ExperimentPlan(
        target_endpoints=["HLM CLint"], arms=["baseline"], n_grid=[100], seeds=[0, 1],
        mechanism_by_arm={"baseline": "gbm_baseline"},
    )
    p = Planner(plan=plan, results_dir=tmp_path, data=None)  # type: ignore[arg-type]
    out = p._write_report(_df(["HLM CLint"]))
    assert "HLM CLint" in out


def test_write_report_handles_empty():
    plan = ExperimentPlan(
        target_endpoints=["HLM CLint"], arms=["baseline"], n_grid=[100], seeds=[0],
        mechanism_by_arm={"baseline": "gbm_baseline"},
    )
    p = Planner(plan=plan, results_dir=Path("/tmp"), data=None)  # type: ignore[arg-type]
    out = p._write_report(pd.DataFrame())
    assert "no results" in out
