"""LLM-driven agents — subclasses of the deterministic agents that delegate the agentic
decision points to Aitta.

Three decisions are LLM-routed (the rest of the pipeline stays deterministic Python):

1. **DataAgent.select_sources** — given the coverage profile and a calibration preview for
   each candidate, decide which external sources to keep.
2. **MLAgent.choose_mechanism** — given the pool shape, choose a training mechanism.
3. **Planner narrative** — given the aggregated results table, write the §0 characterization.

By design these calls happen on the orchestrator, not the SLURM worker. The DataAgent's
decisions get cached on disk by `AittaClient.structured(...)` so re-runs are free; the ML
mechanism choice is similarly cached per (endpoint, arm, n) and baked into the JobSpec.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from data.harmonize import _calibration
from data.registry import ExternalEndpointMap, ExternalSource, sources_for_target

from .contracts import Mechanism
from .data_agent import DataAgent
from .llm import AittaClient
from .ml_agent import MLAgent
from .planner import Planner

log = logging.getLogger(__name__)


# ---- Schemas -------------------------------------------------------------

class _SourceChoice(BaseModel):
    source_name: str
    include: bool
    rationale: str = ""


class SourceSelection(BaseModel):
    decisions: list[_SourceChoice] = Field(default_factory=list)


class MechanismChoice(BaseModel):
    mechanism: Mechanism
    rationale: str = ""


class ReportNarrative(BaseModel):
    summary: str
    per_endpoint: dict[str, str] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)


# ---- DataAgent override --------------------------------------------------

class LLMDataAgent(DataAgent):
    def __init__(self, data, llm: AittaClient):
        super().__init__(data)
        self.llm = llm

    def select_sources(
        self, target_endpoint: str
    ) -> list[tuple[ExternalSource, ExternalEndpointMap]]:
        candidates = sources_for_target(target_endpoint)
        if not candidates:
            return []

        # Build a compact preview the LLM can reason over: match quality, notes,
        # and a fast calibration check on InChIKey overlap.
        previews = []
        for src, m in candidates:
            cal = _quick_calibration(self.data.train, target_endpoint, src, m)
            previews.append({
                "source_name": src.name,
                "source_column": m.source_column,
                "match_quality": m.match_quality,
                "notes": m.notes,
                "calibration": cal,
            })
        coverage = self.data.coverage.to_dict("records")

        decision = self.llm.structured(
            system=(
                "You are the Data agent for an ADMET few-shot transfer study. "
                "Decide which external sources to include for the given target endpoint. "
                "Prefer sources with near-identical assays or good calibration alignment. "
                "Reject sources where the calibration r2 is low AND the match quality is "
                "weak (e.g. different protocol or species). Be willing to include "
                "imperfect sources as auxiliary heads when n_overlap is small."
            ),
            user=json.dumps({
                "target_endpoint": target_endpoint,
                "coverage_profile": coverage,
                "candidates": previews,
            }, indent=2),
            schema=SourceSelection,
            cache_key=f"select_sources::{target_endpoint}",
        )

        keep_names = {d.source_name for d in decision.decisions if d.include}
        log.info("LLM kept %d/%d sources for %s: %s",
                 len(keep_names), len(candidates), target_endpoint, sorted(keep_names))
        return [(s, m) for s, m in candidates if s.name in keep_names]


def _quick_calibration(target_df: pd.DataFrame, target_endpoint: str,
                       src: ExternalSource, m: ExternalEndpointMap) -> dict:
    """Best-effort preview calibration for the LLM. Failures don't block — we report 'unknown'."""
    try:
        src_df = src.loader()
        src_df = src_df.copy()
        src_df["__aux"] = m.harmonize(src_df)
        check = _calibration(target_df, target_endpoint, src_df, "__aux")
        return {
            "n_overlap": check.n_overlap,
            "slope": check.slope,
            "intercept": check.intercept,
            "r2": check.r2,
            "aligned": check.aligned,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---- MLAgent override ----------------------------------------------------

class LLMMLAgent(MLAgent):
    def __init__(self, llm: AittaClient):
        self.llm = llm

    def choose_mechanism(self, pool, requested: Mechanism) -> Mechanism:
        target = pool.target_spec.transformed_column
        n_target = int(pool.df[target].notna().sum())
        per_task = {c: int(pool.df[c].notna().sum()) for c in pool.task_columns}

        decision = self.llm.structured(
            system=(
                "You are the ML agent. Choose a training mechanism for an ADMET multitask "
                "few-shot setup. Rules of thumb:\n"
                "- gbm_baseline: only for the explicit single-task baseline arm.\n"
                "- frozen_embed: best when n_target is very small (<20-30) — fit a light "
                "  head on fingerprints/embeddings rather than train an MPNN.\n"
                "- mt_cotrain: default when n_target is moderate and auxiliary tasks have "
                "  reasonable label counts.\n"
                "- pretrain_finetune: when a strong external source (e.g. near-identical "
                "  assay) has many labels — pretrain on it, then fine-tune on the target.\n"
                "Honor the requested mechanism unless the pool shape contradicts it."
            ),
            user=json.dumps({
                "requested": requested,
                "n_target": n_target,
                "labels_per_task": per_task,
                "target_endpoint": pool.target_endpoint,
                "provenance": pool.provenance,
            }, indent=2),
            schema=MechanismChoice,
            cache_key=f"choose_mechanism::{pool.target_endpoint}::{requested}::{n_target}",
        )
        log.info("LLM picked mechanism=%s for %s (n_target=%d, requested=%s): %s",
                 decision.mechanism, pool.target_endpoint, n_target, requested,
                 decision.rationale)
        return decision.mechanism


# ---- Planner narrative ---------------------------------------------------

class LLMPlanner(Planner):
    def __init__(self, *args, llm: AittaClient, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm

    def _write_report(self, df: pd.DataFrame) -> str:
        templated = super()._write_report(df)
        if df.empty:
            return templated
        agg = (
            df.groupby(["target_endpoint", "arm", "n"], dropna=False)["rae"]
            .agg(["mean", "std", "count"]).reset_index()
        )
        narrative = self.llm.structured(
            system=(
                "You write the characterization deliverable for an ADMET few-shot transfer "
                "study. The brief is: 'which auxiliary tasks transfer, in what data regime, "
                "by how much?' Be specific and quantitative. Mention error bands; do not "
                "claim a lift inside the band. Identify where transfer arms help vs. where "
                "they don't, and propose what to investigate next."
            ),
            user=json.dumps({"aggregate_results": agg.to_dict("records")}, indent=2,
                            default=_jsonable),
            schema=ReportNarrative,
            cache_key=f"narrative::{_hash_df(agg)}",
        )
        out = [templated, "\n## LLM characterization\n", narrative.summary, ""]
        if narrative.per_endpoint:
            out.append("### Per-endpoint\n")
            for ep, txt in narrative.per_endpoint.items():
                out.append(f"**{ep}** — {txt}\n")
        if narrative.open_questions:
            out.append("### Open questions\n")
            for q in narrative.open_questions:
                out.append(f"- {q}")
        return "\n".join(out)


def _hash_df(df: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(df, index=True).sum())


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if pd.isna(o):
        return None
    raise TypeError(f"not json-serializable: {type(o).__name__}")
