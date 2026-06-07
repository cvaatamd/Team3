"""Single-task LightGBM baseline (§3 — the mandatory strong baseline)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .featurizers import featurize

log = logging.getLogger(__name__)


@dataclass
class GBMConfig:
    n_estimators: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 100
    val_fraction: float = 0.15
    seed: int = 0


def train_predict_gbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_transformed_col: str,
    smiles_col: str = "SMILES",
    *,
    config: GBMConfig | None = None,
) -> np.ndarray:
    """Train on the transformed target column; return predictions on test (still transformed)."""
    import lightgbm as lgb

    cfg = config or GBMConfig()
    train_df = train_df.dropna(subset=[target_transformed_col]).copy()
    if train_df.empty:
        raise ValueError(f"no labeled rows for {target_transformed_col}")

    rng = np.random.default_rng(cfg.seed)
    n = len(train_df)
    val_n = max(1, int(np.ceil(cfg.val_fraction * n)))
    perm = rng.permutation(n)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]

    X_all = featurize(train_df[smiles_col].tolist())
    y_all = train_df[target_transformed_col].astype(float).to_numpy()
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_te = featurize(test_df[smiles_col].tolist())

    model = lgb.LGBMRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.seed,
        deterministic=True,
        verbose=-1,
    )
    fit_kwargs: dict = {}
    if cfg.early_stopping_rounds and len(X_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
    model.fit(X_tr, y_tr, **fit_kwargs)
    return model.predict(X_te)
