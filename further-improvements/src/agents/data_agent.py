"""Deterministic Data agent (§5.1).

Given a target endpoint and pool request, return a curated, harmonized training pool plus
a manifest of every decision it made (source selection, calibration, merge strategy, exclusions).

The class shape mirrors what an LLM controller would call into — same input/output, just a
pluggable `select_sources` for the genuinely agentic decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data.expansionrx import (
    EndpointSpec,
    ExpansionRxData,
    flagged_slice_mask,
    subsample_target,
)
from data.harmonize import HarmonizationResult, harmonize_external
from data.registry import sources_for_target
from .contracts import PoolRequest, TrainingPoolManifest

log = logging.getLogger(__name__)


@dataclass
class TrainingPool:
    df: pd.DataFrame
    task_columns: list[str]              # target transformed col first, then auxiliaries
    target_endpoint: str
    target_spec: EndpointSpec
    provenance: dict = field(default_factory=dict)
    harmonization_log: dict = field(default_factory=dict)

    def manifest(self) -> TrainingPoolManifest:
        return TrainingPoolManifest(
            target_endpoint=self.target_endpoint,
            task_columns=list(self.task_columns),
            n_rows=len(self.df),
            n_per_task={c: int(self.df[c].notna().sum()) for c in self.task_columns},
            provenance=self.provenance,
            harmonization_log=self.harmonization_log,
        )


class DataAgent:
    def __init__(self, data: ExpansionRxData):
        self.data = data
        self._spec_by_col = {ep.column: ep for ep in data.endpoints}

    def target_spec(self, endpoint: str) -> EndpointSpec:
        try:
            return self._spec_by_col[endpoint]
        except KeyError as e:
            raise ValueError(f"unknown target endpoint: {endpoint}") from e

    def select_sources(
        self, target_endpoint: str
    ) -> list[tuple]:
        """Agentic choice: which external sources to include for this target.

        Default policy: include all registry entries that map to this target. An LLM-driven
        controller can subclass this method to skip noisy sources based on the calibration
        result or the coverage profile.
        """
        return sources_for_target(target_endpoint)

    def build_pool(self, req: PoolRequest) -> TrainingPool:
        spec = self.target_spec(req.target_endpoint)
        target_col = spec.column
        target_trans = spec.transformed_column

        # 1) start from ExpansionRx train. Optionally drop flagged slices for this endpoint.
        df = self.data.train.copy()
        n_dropped = 0
        if req.exclude_flagged_slices:
            bad = flagged_slice_mask(df, target_col)
            n_dropped = int(bad.sum())
            df = df.loc[~bad].copy()

        # 2) subsample the target label budget to `n`.
        if req.n is not None:
            df = subsample_target(
                df, target_col, n=req.n, seed=req.seed,
                respect_temporal_order=False,
                exclude_flagged=False,  # already done above
            )

        task_cols: list[str] = [target_trans]
        n_per_task = {target_trans: int(df[target_trans].notna().sum())}

        # 3) intra-task auxiliaries: every other ExpansionRx endpoint that has labels in `df`.
        if req.include_intra_task:
            for ep in self.data.endpoints:
                if ep.column == target_col:
                    continue
                col = ep.transformed_column
                if col in df.columns and df[col].notna().any():
                    task_cols.append(col)
                    n_per_task[col] = int(df[col].notna().sum())

        # 4) external sources, harmonized → either co-trained as aux head, or pooled.
        harmonization_log: dict = {}
        if req.include_external:
            for src, mapping in self.select_sources(target_col):
                try:
                    res = harmonize_external(
                        df, spec, src, mapping,
                        shuffle_seed=req.shuffle_external_seed,  # Patch 3: negative control
                    )
                except Exception as e:
                    log.warning("skipping source %s for %s: %s", src.name, target_col, e)
                    harmonization_log[src.name] = {"error": str(e)}
                    continue
                df, n_per_task = self._merge_harmonized(
                    df, res, spec, task_cols, n_per_task
                )
                harmonization_log[src.name] = self._summarize_harmonization(res)

        # Drop rows with no label in ANY task column. In sparse arms (baseline / external) most
        # ExpansionRx rows carry neither the target nor an aux label; under the masked MT loss
        # these produce all-masked batches whose loss is 0/0 = NaN, which propagates and kills
        # the whole model (observed as RAE=nan on MBPB external). Multi-task arms rarely hit this
        # because their extra heads label nearly every row. Keeping only labeled rows is a no-op
        # for training signal and makes every batch well-defined.
        labeled = df[task_cols].notna().any(axis=1)
        n_unlabeled = int((~labeled).sum())
        if n_unlabeled:
            df = df.loc[labeled].copy()

        provenance = {
            "target_endpoint": target_col,
            "n_request": req.n,
            "n_unlabeled_dropped": n_unlabeled,
            "seed": req.seed,
            "exclude_flagged_slices": req.exclude_flagged_slices,
            "n_flagged_dropped": n_dropped,
            "include_intra_task": req.include_intra_task,
            "include_external": req.include_external,
            "n_rows_total": len(df),
        }

        return TrainingPool(
            df=df.reset_index(drop=True),
            task_columns=task_cols,
            target_endpoint=target_col,
            target_spec=spec,
            provenance=provenance,
            harmonization_log=harmonization_log,
        )

    def _merge_harmonized(
        self,
        df: pd.DataFrame,
        res: HarmonizationResult,
        spec: EndpointSpec,
        task_cols: list[str],
        n_per_task: dict[str, int],
    ) -> tuple[pd.DataFrame, dict]:
        """Concatenate harmonized source as either a separate aux head or pooled into target."""
        aux_trans = f"{res.aux_column}__t"
        if res.merge_strategy == "pool":
            # Aligned calibration -> rows go into the target column directly.
            src_df = res.df[["SMILES", res.aux_column, aux_trans]].copy()
            src_df = src_df.rename(columns={
                res.aux_column: spec.column,
                aux_trans: spec.transformed_column,
            })
            for c in df.columns:
                if c not in src_df.columns:
                    src_df[c] = np.nan
            df = pd.concat([df, src_df[df.columns]], ignore_index=True)
            n_per_task[spec.transformed_column] = int(df[spec.transformed_column].notna().sum())
        else:
            # Aux head -> add the source rows with a *separate* labeled column.
            src_df = res.df[["SMILES", res.aux_column, aux_trans]].copy()
            for c in df.columns:
                if c not in src_df.columns:
                    src_df[c] = np.nan
            for c in src_df.columns:
                if c not in df.columns:
                    df[c] = np.nan
            df = pd.concat([df, src_df[df.columns]], ignore_index=True)
            if aux_trans not in task_cols:
                task_cols.append(aux_trans)
            n_per_task[aux_trans] = int(df[aux_trans].notna().sum())
        return df, n_per_task

    @staticmethod
    def _summarize_harmonization(res: HarmonizationResult) -> dict:
        c = res.calibration
        return {
            "source": res.source_name,
            "aux_column": res.aux_column,
            "merge_strategy": res.merge_strategy,
            "calibration": {
                "n_overlap": c.n_overlap,
                "slope": c.slope,
                "intercept": c.intercept,
                "r2": c.r2,
                "aligned": c.aligned,
            },
            "notes": res.notes,
        }
