"""Typed contracts shared by the three agents (§5).

These are deliberately pydantic models so:
- the contracts can be serialized to / from JSON for SLURM hand-off
- an agent can be either a deterministic Python function OR an LLM controller behind the same
  interface
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd  # noqa: F401  (consumers may want it)
from pydantic import BaseModel, ConfigDict, Field

Arm = Literal["baseline", "intra_task", "external", "both"]
Mechanism = Literal["mt_cotrain", "pretrain_finetune", "frozen_embed", "gbm_baseline"]


class PoolRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_endpoint: str
    n: Optional[int] = None
    include_intra_task: bool = False
    include_external: bool = False
    seed: int = 0
    exclude_flagged_slices: bool = True
    # Patch 3 (further-improvements): negative control. When set, external source labels are
    # permuted before merging. None -> original behaviour.
    shuffle_external_seed: Optional[int] = None


class TrainingPoolManifest(BaseModel):
    """Serializable view of a TrainingPool (provenance/log only — the df travels separately)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_endpoint: str
    task_columns: list[str]
    n_rows: int
    n_per_task: dict[str, int]
    provenance: dict[str, Any] = Field(default_factory=dict)
    harmonization_log: dict[str, Any] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    """Request to the ML agent. The pool df is passed alongside (in-memory or via parquet path)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_endpoint: str
    arm: Arm
    n: Optional[int]
    seed: int
    mechanism: Mechanism
    pool_manifest: TrainingPoolManifest
    pool_parquet: Optional[str] = None       # parquet path when crossing process boundary
    test_parquet: Optional[str] = None       # parquet path for the held-out test
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    target_endpoint: str
    arm: Arm
    n: Optional[int]
    seed: int
    mechanism: Mechanism
    rae: float
    extra_metrics: dict[str, float] = Field(default_factory=dict)
    model_manifest: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class JobSpec(BaseModel):
    """One row of the sweep — what the SLURM dispatcher hands a worker process."""
    job_id: str                                  # stable id for filenames / array idx mapping
    target_endpoint: str
    arm: Arm
    n: Optional[int]
    seed: int
    mechanism: Mechanism
    exclude_flagged_slices: bool = True
    rae_definition: str = "range_normalized"
    # Patch 3 (further-improvements): negative-control toggle, propagated to the data agent.
    shuffle_external_seed: Optional[int] = None


class ExperimentPlan(BaseModel):
    target_endpoints: list[str]
    arms: list[Arm]
    n_grid: list[Optional[int]]
    seeds: list[int]
    mechanism_by_arm: dict[str, Mechanism]
    exclude_flagged_slices: bool = True
    rae_definition: str = "range_normalized"
    # Patch 3 (further-improvements): set in the plan YAML to make this whole sweep a
    # negative-control run (external labels permuted with this seed).
    shuffle_external_seed: Optional[int] = None

    def enumerate_jobs(self) -> list[JobSpec]:
        out: list[JobSpec] = []
        for ep in self.target_endpoints:
            for arm in self.arms:
                mech = self.mechanism_by_arm[arm]
                for n in self.n_grid:
                    for seed in self.seeds:
                        n_tag = "full" if n is None else str(n)
                        job_id = f"{_slug(ep)}__{arm}__{n_tag}__s{seed}"
                        out.append(JobSpec(
                            job_id=job_id,
                            target_endpoint=ep,
                            arm=arm,
                            n=n,
                            seed=seed,
                            mechanism=mech,
                            exclude_flagged_slices=self.exclude_flagged_slices,
                            rae_definition=self.rae_definition,
                            shuffle_external_seed=self.shuffle_external_seed,
                        ))
        return out


def _slug(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace(">", "to")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
    )
