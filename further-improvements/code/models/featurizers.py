"""ECFP4 (2048-bit) + RDKit descriptor featurizer — the input to the GBM baseline (§3)."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

ECFP_RADIUS = 2  # ECFP4 = radius 2 in RDKit speak
ECFP_BITS = 2048


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        return Chem, AllChem, Descriptors
    except ImportError as e:
        raise ImportError(
            "rdkit not installed — install the `chem` extra: pip install -e '.[chem]'"
        ) from e


def ecfp4(smiles_iter: Iterable[str]) -> np.ndarray:
    Chem, AllChem, _ = _require_rdkit()
    fps = np.zeros((0, ECFP_BITS), dtype=np.uint8)
    out = []
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            out.append(np.zeros(ECFP_BITS, dtype=np.uint8))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_BITS)
        arr = np.zeros(ECFP_BITS, dtype=np.uint8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        out.append(arr)
    return np.stack(out) if out else fps


def rdkit_descriptors(smiles_iter: Iterable[str], names: list[str] | None = None) -> np.ndarray:
    """A compact, well-behaved subset of RDKit descriptors. Avoid the full ~200 — many are noisy."""
    Chem, _, Descriptors = _require_rdkit()
    default_names = [
        "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
        "NumRotatableBonds", "NumAromaticRings", "RingCount", "FractionCSP3",
        "HeavyAtomCount", "NumHeteroatoms", "LabuteASA", "BertzCT",
    ]
    names = names or default_names
    fns = [getattr(Descriptors, n) for n in names]
    rows = []
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            rows.append([np.nan] * len(fns))
            continue
        rows.append([f(mol) for f in fns])
    arr = np.asarray(rows, dtype=float)
    # impute NaNs with column means (RDKit can fail on weird inputs)
    col_mean = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_mean, inds[1])
    return arr


def featurize(smiles_iter: Iterable[str]) -> np.ndarray:
    """Concatenated ECFP4 + descriptors — the GBM baseline input."""
    smiles_list = list(smiles_iter)
    ecfp = ecfp4(smiles_list)
    desc = rdkit_descriptors(smiles_list)
    return np.concatenate([ecfp.astype(np.float32), desc.astype(np.float32)], axis=1)
