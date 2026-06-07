"""SMILES canonicalization, salt stripping, neutralization, deduplication.

RDKit is optional at import time so the harness can still be imported on a node where it's
not yet installed; the actual standardization functions raise a clear error if RDKit is missing.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        return Chem, rdMolStandardize
    except ImportError as e:
        raise ImportError(
            "rdkit not installed — add the `chem` extra: pip install -e '.[chem]'"
        ) from e


def canonical_smiles(smiles: str) -> str | None:
    Chem, _ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def standardize_one(smiles: str) -> str | None:
    """Canonical + parent (salt strip) + neutralized SMILES, or None if RDKit can't parse."""
    Chem, rdMS = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMS.ChargeParent(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def standardize_frame(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    *,
    drop_failed: bool = True,
    dedupe_on_inchikey: bool = True,
) -> pd.DataFrame:
    Chem, _ = _require_rdkit()
    df = df.copy()
    std = df[smiles_col].map(standardize_one)
    df["smiles_std"] = std
    if drop_failed:
        df = df.loc[df["smiles_std"].notna()].copy()
    if dedupe_on_inchikey:
        df["inchikey"] = df["smiles_std"].map(
            lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if s else None
        )
        df = df.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    return df


def inchikeys(smiles: Iterable[str]) -> list[str | None]:
    Chem, _ = _require_rdkit()
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(Chem.MolToInchiKey(mol) if mol is not None else None)
    return out
