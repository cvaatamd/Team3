"""Frozen-embedding + light head (§3).

For tiny `n` the cheapest credible model: take fingerprint/descriptor features (or pretrained
Chemprop embeddings if a checkpoint is provided) and fit a ridge or GBM head per task.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .featurizers import featurize

log = logging.getLogger(__name__)


@dataclass
class FrozenEmbedConfig:
    head: str = "ridge"           # "ridge" | "gbm"
    alpha: float = 1.0            # ridge regularization
    seed: int = 0
    chemprop_checkpoint: str | None = None  # if set, use chemprop embeddings


def _chemprop_embeddings(smiles: list[str], checkpoint: str) -> np.ndarray:
    """Pull the MPNN-encoded vector for each molecule from a chemprop v2 checkpoint."""
    import torch
    from chemprop import data as cp_data
    from chemprop import featurizers, models
    from torch.utils.data import DataLoader

    mpnn = models.MPNN.load_from_checkpoint(checkpoint)
    mpnn.eval()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dps = [cp_data.MoleculeDatapoint.from_smi(s) for s in smiles]
    ds = cp_data.MoleculeDataset(dps, featurizer=featurizer)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=cp_data.collate_batch)
    out = []
    with torch.no_grad():
        for batch in loader:
            h = mpnn.fingerprint(batch.bmg)
            out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


def train_predict_frozen(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_transformed_col: str,
    *,
    smiles_col: str = "SMILES",
    config: FrozenEmbedConfig | None = None,
) -> np.ndarray:
    cfg = config or FrozenEmbedConfig()
    train_df = train_df.dropna(subset=[target_transformed_col])
    if train_df.empty:
        raise ValueError(f"no labeled rows for {target_transformed_col}")

    if cfg.chemprop_checkpoint:
        X_tr = _chemprop_embeddings(train_df[smiles_col].tolist(), cfg.chemprop_checkpoint)
        X_te = _chemprop_embeddings(test_df[smiles_col].tolist(), cfg.chemprop_checkpoint)
    else:
        X_tr = featurize(train_df[smiles_col].tolist())
        X_te = featurize(test_df[smiles_col].tolist())
    y_tr = train_df[target_transformed_col].astype(float).to_numpy()

    if cfg.head == "ridge":
        model = Ridge(alpha=cfg.alpha, random_state=cfg.seed)
        model.fit(X_tr, y_tr)
        return model.predict(X_te)
    if cfg.head == "gbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=3, random_state=cfg.seed, verbose=-1,
        )
        model.fit(X_tr, y_tr)
        return model.predict(X_te)
    raise ValueError(f"unknown head: {cfg.head}")
