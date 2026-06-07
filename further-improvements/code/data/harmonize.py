"""Harmonize an external source onto an ExpansionRx target (§2.4).

Steps:
1. Load source, project to target raw scale via the registry mapping.
2. Calibration check via InChIKey overlap with the ExpansionRx target frame.
3. Merge strategy decision: default to a *separate auxiliary head* unless calibration is aligned.
4. Standardize SMILES + dedupe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .registry import ExternalEndpointMap, ExternalSource, sources_for_target
from .standardize import inchikeys, standardize_frame
from .expansionrx import EndpointSpec, _forward

log = logging.getLogger(__name__)

MergeStrategy = Literal["aux_head", "pool"]


@dataclass
class CalibrationCheck:
    n_overlap: int
    slope: float | None = None
    intercept: float | None = None
    r2: float | None = None
    aligned: bool = False


@dataclass
class HarmonizationResult:
    df: pd.DataFrame                # SMILES + harmonized target column (raw scale) + transformed col
    source_name: str
    target_endpoint: str
    aux_column: str                 # name of the harmonized column in `df`
    calibration: CalibrationCheck
    merge_strategy: MergeStrategy
    notes: list[str] = field(default_factory=list)


def _calibration(
    target_df: pd.DataFrame,
    target_endpoint: str,
    aux_df: pd.DataFrame,
    aux_column: str,
    *,
    align_threshold_r2: float = 0.7,
    align_slope_band: tuple[float, float] = (0.7, 1.3),
) -> CalibrationCheck:
    """Regress source vs target on InChIKey overlap. Aligned ≈ slope≈1, intercept≈0, r2 high."""
    if "inchikey" not in target_df.columns:
        target_df = target_df.copy()
        target_df["inchikey"] = inchikeys(target_df["SMILES"])
    if "inchikey" not in aux_df.columns:
        aux_df = aux_df.copy()
        aux_df["inchikey"] = inchikeys(aux_df["SMILES"])

    merged = target_df[["inchikey", target_endpoint]].merge(
        aux_df[["inchikey", aux_column]], on="inchikey", how="inner"
    ).dropna()
    n = len(merged)
    if n < 5:
        return CalibrationCheck(n_overlap=n)
    x = merged[aux_column].astype(float).values
    y = merged[target_endpoint].astype(float).values
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    aligned = (
        r2 >= align_threshold_r2
        and align_slope_band[0] <= slope <= align_slope_band[1]
        and abs(intercept) < (np.abs(y).mean() * 0.1 + 1e-6)
    )
    return CalibrationCheck(n_overlap=n, slope=float(slope), intercept=float(intercept),
                            r2=float(r2), aligned=bool(aligned))


def harmonize_external(
    target_df: pd.DataFrame,
    target_spec: EndpointSpec,
    source: ExternalSource,
    mapping: ExternalEndpointMap,
    *,
    force_strategy: MergeStrategy | None = None,
    standardize: bool = True,
    shuffle_seed: int | None = None,
) -> HarmonizationResult:
    """Project a source onto target raw scale; choose merge strategy via calibration.

    Patch 3 (further-improvements): when `shuffle_seed` is set, the harmonized source labels are
    randomly permuted across molecules (keeping the marginal distribution but destroying the
    structure-activity link). This is the NEGATIVE CONTROL: if the real external arm beats this
    label-shuffled arm, the lift is genuine transferred signal, not mere extra-column regularization.
    """
    src = source.loader()
    if "SMILES" not in src.columns:
        raise ValueError(f"source {source.name} has no SMILES column")
    src = src.copy()
    src["__src_aux"] = mapping.harmonize(src)
    aux_col = f"{mapping.target_endpoint} [{source.name}]"
    src = src.rename(columns={"__src_aux": aux_col})
    src = src[["SMILES", aux_col]].dropna()
    if shuffle_seed is not None:
        permuted = src[aux_col].sample(frac=1.0, random_state=shuffle_seed).to_numpy()
        src[aux_col] = permuted
    if standardize:
        src = standardize_frame(src, smiles_col="SMILES", drop_failed=True,
                                dedupe_on_inchikey=True)

    cal = _calibration(target_df, target_spec.column, src, aux_col)
    if force_strategy is not None:
        strategy = force_strategy
    else:
        strategy = "pool" if cal.aligned else "aux_head"

    # Add the transformed column under the target's transform so it shares the MT head's space.
    transformed_col = f"{aux_col}__t"
    src[transformed_col] = _forward(src[aux_col], target_spec)

    notes = [
        f"source={source.name}",
        f"match_quality={mapping.match_quality}",
        f"calibration: n_overlap={cal.n_overlap}, slope={cal.slope}, "
        f"intercept={cal.intercept}, r2={cal.r2}, aligned={cal.aligned}",
        f"merge_strategy={strategy}",
    ]
    if shuffle_seed is not None:
        notes.append(f"NEGATIVE_CONTROL=labels_shuffled(seed={shuffle_seed})")
    if mapping.notes:
        notes.append(f"map_notes={mapping.notes}")
    log.info("\n".join(notes))

    return HarmonizationResult(
        df=src,
        source_name=source.name,
        target_endpoint=target_spec.column,
        aux_column=aux_col,
        calibration=cal,
        merge_strategy=strategy,
        notes=notes,
    )


def harmonize_for_target(
    target_df: pd.DataFrame,
    target_spec: EndpointSpec,
) -> list[HarmonizationResult]:
    """Apply every registry mapping for this target. Returns one result per source."""
    out: list[HarmonizationResult] = []
    for src, mapping in sources_for_target(target_spec.column):
        try:
            out.append(harmonize_external(target_df, target_spec, src, mapping))
        except Exception as e:  # source-specific failures (network, missing dep) shouldn't kill all
            log.warning("harmonization failed for %s -> %s: %s", src.name, target_spec.column, e)
    return out
