"""Curated registry of external ADMET sources (§2.3).

The data agent draws from this registry, not the open web. Each entry knows:
- its loader (a callable returning a DataFrame with SMILES + the source endpoint columns)
- the mapping from source columns to ExpansionRx target endpoints, with the harmonization
  transform that aligns the source to the target's *raw* scale
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ExternalEndpointMap:
    target_endpoint: str          # e.g. "HLM CLint"
    source_column: str            # column in source frame
    harmonize: Callable[[pd.DataFrame], pd.Series]  # source row -> value on target raw scale
    match_quality: str            # "near-identical" / "strong" / "analogous" / "species-transfer" / ...
    notes: str = ""


@dataclass
class ExternalSource:
    name: str                     # e.g. "biogen/adme-fang-v1"
    loader: Callable[[], pd.DataFrame]
    mappings: list[ExternalEndpointMap] = field(default_factory=list)


# ----- Loaders --------------------------------------------------------------

def load_biogen_fang() -> pd.DataFrame:
    import polaris as po
    ds = po.load_dataset("biogen/adme-fang-v1")
    df = ds.table if hasattr(ds, "table") else ds.to_pandas()
    return df


def load_tdc_caco2() -> pd.DataFrame:
    from tdc.single_pred import ADME
    data = ADME(name="Caco2_Wang")
    df = data.get_data()  # cols: Drug_ID, Drug (SMILES), Y
    return df.rename(columns={"Drug": "SMILES"})


# ----- Harmonization closures ----------------------------------------------

def _hlm_clint_from_log_mlmin_kg(df: pd.DataFrame) -> pd.Series:
    # Biogen: LOG HLM_CLint (mL/min/kg). ExpansionRx HLM CLint is raw mL/min/kg.
    return np.power(10.0, df["LOG HLM_CLint (mL/min/kg)"].astype(float))


def _rlm_clint_from_log(df: pd.DataFrame) -> pd.Series:
    return np.power(10.0, df["LOG RLM_CLint (mL/min/kg)"].astype(float))


def _ksol_um_from_log_ugml(df: pd.DataFrame) -> pd.Series:
    """Biogen LOG SOLUBILITY PH 6.8 (ug/mL) → µM via molecular weight (§2.4)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    ug_per_ml = np.power(10.0, df["LOG SOLUBILITY PH 6.8 (ug/mL)"].astype(float))
    mws = []
    for smi in df["SMILES"]:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        mws.append(Descriptors.MolWt(m) if m is not None else np.nan)
    mw = pd.Series(mws, index=df.index)
    # µM = µg/mL ÷ MW × 1000
    return ug_per_ml / mw * 1000.0


def _mdr1_efflux_from_log(df: pd.DataFrame) -> pd.Series:
    return np.power(10.0, df["LOG MDR1-MDCK ER"].astype(float))


def _ppb_pct_from_human(df: pd.DataFrame) -> pd.Series:
    # Biogen hPPB stored as fraction unbound; convert to % unbound to match ExpansionRx MPPB.
    return df["hPPB"].astype(float) * 100.0


def _caco2_papp_from_tdc(df: pd.DataFrame) -> pd.Series:
    # TDC Caco2_Wang Y is log10 of Papp (cm/s); ExpansionRx Papp is 1e-6 cm/s.
    return np.power(10.0, df["Y"].astype(float)) * 1e6


# ----- Registry --------------------------------------------------------------

REGISTRY: dict[str, ExternalSource] = {
    "biogen/adme-fang-v1": ExternalSource(
        name="biogen/adme-fang-v1",
        loader=load_biogen_fang,
        mappings=[
            ExternalEndpointMap(
                target_endpoint="HLM CLint",
                source_column="LOG HLM_CLint (mL/min/kg)",
                harmonize=_hlm_clint_from_log_mlmin_kg,
                match_quality="near-identical",
                notes="same assay, same units, source is already log10",
            ),
            ExternalEndpointMap(
                target_endpoint="RLM CLint",
                source_column="LOG RLM_CLint (mL/min/kg)",
                harmonize=_rlm_clint_from_log,
                match_quality="strong",
            ),
            ExternalEndpointMap(
                target_endpoint="KSOL",
                source_column="LOG SOLUBILITY PH 6.8 (ug/mL)",
                harmonize=_ksol_um_from_log_ugml,
                match_quality="protocol-caveat",
                notes="needs ug/mL -> uM via MW; protocol differs",
            ),
            ExternalEndpointMap(
                target_endpoint="Caco-2 Permeability Efflux",
                source_column="LOG MDR1-MDCK ER",
                harmonize=_mdr1_efflux_from_log,
                match_quality="analogous",
                notes="different cell line (MDR1-MDCK vs Caco-2)",
            ),
            ExternalEndpointMap(
                target_endpoint="MPPB",
                source_column="hPPB",
                harmonize=_ppb_pct_from_human,
                match_quality="species-transfer",
                notes="human -> mouse",
            ),
        ],
    ),
    "tdc/Caco2_Wang": ExternalSource(
        name="tdc/Caco2_Wang",
        loader=load_tdc_caco2,
        mappings=[
            ExternalEndpointMap(
                target_endpoint="Caco-2 Permeability Papp A>B",
                source_column="Y",
                harmonize=_caco2_papp_from_tdc,
                match_quality="different-system",
            ),
        ],
    ),
}


def sources_for_target(target_endpoint: str) -> list[tuple[ExternalSource, ExternalEndpointMap]]:
    out = []
    for src in REGISTRY.values():
        for m in src.mappings:
            if m.target_endpoint == target_endpoint:
                out.append((src, m))
    return out
