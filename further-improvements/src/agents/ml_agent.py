"""Deterministic ML agent (§5.2).

Given a training pool and the fixed test split, choose a mechanism, train, evaluate, and return
metrics. Agentic flexibility lives in `choose_mechanism` — defaults are sensible per Plan.md and
an LLM-driven subclass can override.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data.expansionrx import EndpointSpec, invert_transform
from eval.metrics import diagnostics
from models.gbm_baseline import GBMConfig, train_predict_gbm
from .contracts import Arm, EvalResult, Mechanism, _slug
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
    # native scale so any RAE definition can be re-scored offline and the bootstrap can run at
    # the molecule level — no retraining. None == legacy behaviour (scalars only).
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
            # Real transfer (further-improvements): pretrain the MPNN on the auxiliary tasks
            # (intra-task and/or harmonized external columns), then fine-tune on the target.
            ckpt = spec.pool.provenance.get("pretrained_checkpoint") \
                or self._pretrain_aux_checkpoint(spec)
            freeze = os.environ.get("ADMET_FREEZE_MPNN", "0") == "1"
            if ckpt is None:
                log.info("pretrain_finetune: no aux tasks to pretrain on -> from-scratch MT")
            y_pred_t = self._chemprop_mt(spec, pretrained=ckpt, freeze=freeze)
        elif mechanism == "frozen_embed":
            # Few-shot transfer: freeze a pretrained encoder, fit a light head on its embeddings.
            # Falls back to fingerprint features when there is no aux source to pretrain on.
            ckpt = self._pretrain_aux_checkpoint(spec)
            y_pred_t = self._frozen_embed(spec, chemprop_checkpoint=ckpt)
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
        # re-scoring (official RAE) and molecule-level bootstrapping.
        self._persist_preds(spec, y_true, y_pred, mechanism, rng)

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

    @staticmethod
    def _persist_preds(spec: TrainSpec, y_true: np.ndarray, y_pred: np.ndarray,
                       mechanism: Mechanism, test_range: float) -> None:
        """Write one parquet of native-scale (y_true, y_pred) per run, if `preds_dir` is set.

        File name mirrors the run JSON convention so a row maps 1:1 to a sweep cell:
            <endpoint_slug>__<arm>__n<n>__s<seed>.parquet
        """
        if not spec.preds_dir:
            return
        from pathlib import Path

        out_dir = Path(spec.preds_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_tag = "full" if spec.n is None else str(spec.n)
        stem = spec.job_id or (
            f"{_slug(spec.pool.target_endpoint)}__{spec.arm}__n{n_tag}__s{spec.seed}"
        )
        df = pd.DataFrame({"y_true": np.asarray(y_true, dtype=float),
                           "y_pred": np.asarray(y_pred, dtype=float)})
        # Stamp the cell identity + native test range so the re-score step is self-contained.
        df.attrs.update({
            "endpoint": spec.pool.target_endpoint, "arm": spec.arm,
            "n": n_tag, "seed": spec.seed, "mechanism": mechanism,
            "test_range": test_range,
        })
        meta = pd.DataFrame([{
            "endpoint": spec.pool.target_endpoint, "arm": spec.arm, "n": n_tag,
            "seed": spec.seed, "mechanism": mechanism, "test_range": test_range,
        }])
        df.to_parquet(out_dir / f"{stem}.parquet")
        meta.to_parquet(out_dir / f"{stem}.meta.parquet")

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

    def _frozen_embed(self, spec: TrainSpec, *, chemprop_checkpoint: Optional[str] = None) -> np.ndarray:
        from models.frozen_embed import FrozenEmbedConfig, train_predict_frozen
        cfg = FrozenEmbedConfig(seed=spec.seed, chemprop_checkpoint=chemprop_checkpoint)
        return train_predict_frozen(
            spec.pool.df, spec.test_df,
            target_transformed_col=spec.pool.target_spec.transformed_column,
            config=cfg,
        )

    # ---- transfer: pretrain on auxiliary tasks (further-improvements) ----

    def _pretrain_aux_checkpoint(self, spec: TrainSpec) -> Optional[str]:
        """Phase 1 of transfer: pretrain the MPNN on the pool's auxiliary tasks and save a
        checkpoint. Returns the checkpoint path, or None when there is nothing to pretrain on
        (e.g. the single-task baseline arm) so the caller falls back to a from-scratch model.

        Self-contained: it reuses the rows/columns already assembled by the Data agent —
        `task_columns` minus the target are the intra-task and/or harmonized-external heads.
        """
        pool = spec.pool
        target = pool.target_spec.transformed_column
        aux_cols = [c for c in pool.task_columns if c != target]
        if not aux_cols:
            return None
        # Pretrain over the SAME multitask head shape used at fine-tune time (the full pool task
        # set), but on rows that carry an auxiliary label. Keeping n_tasks identical to the
        # fine-tune model means the checkpoint's predictor/metrics line up when we reload it —
        # otherwise chemprop's validation metric raises a task-dimension mismatch. The target
        # column is mostly NaN on these rows and is masked, so the encoder learns from the aux
        # signal exactly as intended.
        task_columns = list(pool.task_columns)
        pre_df = pool.df[pool.df[aux_cols].notna().any(axis=1)].reset_index(drop=True)
        if len(pre_df) < 10:
            log.info("pretrain: only %d aux-labeled rows for %s/%s — skipping pretrain",
                     len(pre_df), pool.target_endpoint, spec.arm)
            return None

        from pathlib import Path
        import tempfile
        if spec.preds_dir:
            ck_dir = Path(spec.preds_dir).parent / "pretrain_ckpts"
        else:
            ck_dir = Path(tempfile.mkdtemp(prefix="pretrain_"))
        ck_dir.mkdir(parents=True, exist_ok=True)
        n_tag = "full" if spec.n is None else str(spec.n)
        stem = spec.job_id or f"{_slug(pool.target_endpoint)}__{spec.arm}__n{n_tag}__s{spec.seed}"
        ckpt = ck_dir / f"{stem}.ckpt"

        from models.chemprop_mt import ChempropConfig, train_predict_chemprop_mt
        # Pretrain epochs kept below the fine-tune budget to bound cost; predictions are ignored,
        # we only want the trained encoder persisted to `checkpoint_out`.
        cfg = ChempropConfig(seed=spec.seed, epochs=50)
        log.info("pretrain: %s/%s on %d aux rows over tasks=%s (aux=%s) -> %s",
                 pool.target_endpoint, spec.arm, len(pre_df), task_columns, aux_cols, ckpt)
        train_predict_chemprop_mt(
            train_df=pre_df, test_df=spec.test_df,
            task_columns=task_columns, target_col=target,
            config=cfg, checkpoint_out=ckpt,
        )
        return str(ckpt)
