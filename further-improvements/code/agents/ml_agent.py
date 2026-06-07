"""Deterministic ML agent (§5.2).

Given a training pool and the fixed test split, choose a mechanism, train, evaluate, and return
metrics. Agentic flexibility lives in `choose_mechanism` — defaults are sensible per Plan.md and
an LLM-driven subclass can override.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data.expansionrx import EndpointSpec, invert_transform
from eval.metrics import diagnostics
from models.gbm_baseline import GBMConfig, train_predict_gbm
from .contracts import Arm, EvalResult, Mechanism
from .data_agent import TrainingPool

log = logging.getLogger(__name__)


@dataclass
class TrainSpec:
    pool: TrainingPool
    test_df: pd.DataFrame
    arm: Arm
    n: Optional[int]
    seed: int
    mechanism: Mechanism
    rae_definition: str = "range_normalized"
    # Patch 1 (further-improvements): when set, persist per-molecule (y_true, y_pred) on the
    # native scale so any RAE definition (e.g. the official one) and a molecule-level bootstrap
    # can be recomputed offline with no retraining. None -> behaves exactly as the original.
    preds_dir: Optional[str] = None
    job_id: Optional[str] = None


class MLAgent:
    def choose_mechanism(self, pool: TrainingPool, requested: Mechanism) -> Mechanism:
        """Default: honor the request. Override to pick e.g. frozen_embed at tiny n."""
        n_target = int(
            pool.df[pool.target_spec.transformed_column].notna().sum()
        )
        if requested == "mt_cotrain" and n_target < 20:
            log.info("auto-downgrade mt_cotrain -> frozen_embed at n_target=%d", n_target)
            return "frozen_embed"
        return requested

    def run(self, spec: TrainSpec) -> EvalResult:
        mechanism = self.choose_mechanism(spec.pool, spec.mechanism)
        t0 = time.time()
        if mechanism == "gbm_baseline":
            y_pred_t = self._gbm(spec)
        elif mechanism == "mt_cotrain":
            y_pred_t = self._chemprop_mt(spec, pretrained=None, freeze=False)
        elif mechanism == "pretrain_finetune":
            ckpt = spec.pool.provenance.get("pretrained_checkpoint")
            y_pred_t = self._chemprop_mt(spec, pretrained=ckpt, freeze=False)
        elif mechanism == "frozen_embed":
            y_pred_t = self._frozen_embed(spec)
        else:
            raise ValueError(f"unknown mechanism: {mechanism}")
        elapsed = time.time() - t0

        spec_t = spec.pool.target_spec
        y_pred = invert_transform(y_pred_t, spec_t)
        y_true = spec.test_df[spec_t.column].to_numpy(dtype=float)

        # Test-set range is the *native scale* dynamic range (§1).
        rng = float(np.nanmax(y_true) - np.nanmin(y_true))
        diag = diagnostics(
            y_true, y_pred,
            rae_definition=spec.rae_definition,  # type: ignore[arg-type]
            test_range=rng,
        )

        # Patch 1 (further-improvements): persist native-scale predictions for free offline
        # re-scoring (official RAE) and molecule-level bootstrap CIs. No-op when preds_dir is None.
        if spec.preds_dir is not None:
            self._persist_predictions(spec, mechanism, y_true, y_pred, test_range=rng)
        return EvalResult(
            target_endpoint=spec.pool.target_endpoint,
            arm=spec.arm,
            n=spec.n,
            seed=spec.seed,
            mechanism=mechanism,
            rae=diag.rae,
            extra_metrics={"mae": diag.mae, "rmse": diag.rmse, "r2": diag.r2,
                           "spearman": diag.spearman, "n_test": diag.n,
                           "elapsed_s": elapsed},
            model_manifest={"mechanism": mechanism},
            provenance=spec.pool.provenance,
        )

    # ---- prediction persistence (Patch 1) ----

    def _persist_predictions(
        self, spec: TrainSpec, mechanism: Mechanism,
        y_true: np.ndarray, y_pred: np.ndarray, *, test_range: float,
    ) -> None:
        """Write native-scale (y_true, y_pred) for this run so RAE can be recomputed offline."""
        import os
        from pathlib import Path

        d = Path(spec.preds_dir)
        d.mkdir(parents=True, exist_ok=True)
        n_tag = "full" if spec.n is None else str(spec.n)
        jid = spec.job_id or f"{spec.pool.target_endpoint}__{spec.arm}__{n_tag}__s{spec.seed}"
        out = pd.DataFrame({"y_true": np.asarray(y_true, float),
                            "y_pred": np.asarray(y_pred, float)})
        smiles_col = spec.pool.target_spec.__dict__.get("smiles_col", "SMILES")
        if smiles_col in spec.test_df.columns and len(spec.test_df) == len(out):
            out.insert(0, "SMILES", spec.test_df[smiles_col].to_numpy())
        # Carry the metadata needed for re-scoring + grouping, so each file is self-describing.
        out.attrs.update({
            "endpoint": spec.pool.target_endpoint, "arm": spec.arm, "n": n_tag,
            "seed": spec.seed, "mechanism": mechanism, "test_range": test_range,
        })
        path = d / f"{_safe(jid)}.parquet"
        out.to_parquet(path)
        # Also stamp metadata into a sidecar JSON (parquet attrs don't always round-trip).
        import json
        (d / f"{_safe(jid)}.meta.json").write_text(json.dumps({
            "endpoint": spec.pool.target_endpoint, "arm": spec.arm, "n": n_tag,
            "seed": spec.seed, "mechanism": mechanism, "test_range": test_range,
            "n_test": int(np.isfinite(y_true).sum()),
        }, indent=2))
        log.info("[preds] wrote %d rows -> %s", len(out), os.fspath(path))

    # ---- mechanism dispatchers ----

    def _gbm(self, spec: TrainSpec) -> np.ndarray:
        tcol = spec.pool.target_spec.transformed_column
        return train_predict_gbm(
            spec.pool.df, spec.test_df, target_transformed_col=tcol,
            config=GBMConfig(seed=spec.seed),
        )

    def _chemprop_mt(
        self, spec: TrainSpec, *, pretrained: Optional[str], freeze: bool
    ) -> np.ndarray:
        # Lazy import so the harness loads on machines without chemprop.
        from models.chemprop_mt import ChempropConfig, train_predict_chemprop_mt
        cfg = ChempropConfig(
            seed=spec.seed, pretrained_checkpoint=pretrained, freeze_mpnn=freeze,
        )
        return train_predict_chemprop_mt(
            train_df=spec.pool.df,
            test_df=spec.test_df,
            task_columns=spec.pool.task_columns,
            target_col=spec.pool.target_spec.transformed_column,
            config=cfg,
        )

    def _frozen_embed(self, spec: TrainSpec) -> np.ndarray:
        from models.frozen_embed import FrozenEmbedConfig, train_predict_frozen
        cfg = FrozenEmbedConfig(seed=spec.seed)
        return train_predict_frozen(
            spec.pool.df, spec.test_df,
            target_transformed_col=spec.pool.target_spec.transformed_column,
            config=cfg,
        )


def _safe(s: str) -> str:
    return (
        s.replace(" ", "_").replace(">", "to").replace("/", "_")
         .replace("(", "").replace(")", "")
    )
