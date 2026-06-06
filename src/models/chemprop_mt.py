"""Chemprop v2 multitask wrapper (§3).

Capabilities we depend on:
- multitask regression with masked loss over NaN targets (per-molecule mask)
- pretrain → fine-tune via loading a checkpoint with optional frozen MPNN

The chemprop v2.x API has moved between minor versions — Plan.md §3 calls this out. We import
inside functions so the rest of the harness imports cleanly without chemprop installed; the
functions raise a clear error if it's missing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ChempropConfig:
    epochs: int = 80
    batch_size: int = 64
    init_lr: float = 1e-4
    # max_lr=1e-3 caused NaN divergence on low-task pools (e.g. MBPB external: target + one
    # weak aux head). Lowered to 2e-4 to keep the Noam schedule's peak step small enough to
    # avoid the exploding-gradient blow-up while still converging within `epochs`.
    max_lr: float = 2e-4
    final_lr: float = 1e-4
    depth: int = 3
    hidden_size: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
    seed: int = 0
    freeze_mpnn: bool = False
    pretrained_checkpoint: str | None = None
    devices: int = 1
    accelerator: str = "auto"
    grad_clip: float = 1.0  # clip exploding gradients -> avoids NaN divergence on sparse tasks
    val_fraction: float = 0.1
    smiles_col: str = "SMILES"
    extra_columns_to_ignore: list[str] = field(default_factory=list)


def _import_chemprop():
    try:
        import chemprop  # noqa: F401
        from chemprop import data as cp_data
        from chemprop import featurizers, models, nn
        from lightning import pytorch as pl
        from torch.utils.data import DataLoader
        return cp_data, featurizers, models, nn, pl, DataLoader
    except ImportError as e:
        raise ImportError(
            "chemprop / lightning not installed — add the `chem` extra: pip install -e '.[chem]'"
        ) from e


def _build_datasets(train_df, val_df, test_df, smiles_col, task_columns, cp_data, featurizers):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def to_dataset(df, has_y=True):
        smis = df[smiles_col].tolist()
        if has_y:
            Y = df[task_columns].to_numpy(dtype=float)  # NaN = masked
        else:
            Y = np.zeros((len(df), len(task_columns)), dtype=float)
        data = [cp_data.MoleculeDatapoint.from_smi(s, y) for s, y in zip(smis, Y)]
        return cp_data.MoleculeDataset(data, featurizer=featurizer)

    return to_dataset(train_df), to_dataset(val_df), to_dataset(test_df, has_y=False)


def train_predict_chemprop_mt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_columns: list[str],
    target_col: str,
    *,
    config: ChempropConfig | None = None,
    checkpoint_out: Path | None = None,
) -> np.ndarray:
    """Train a multitask Chemprop model, return predictions on test for `target_col`.

    `task_columns[0]` is conventionally the target — we expose all in the head and pick the
    column at predict time. NaN labels are masked by chemprop's built-in regression loss.
    """
    cp_data, featurizers, models, nn, pl, DataLoader = _import_chemprop()
    cfg = config or ChempropConfig()
    rng = np.random.default_rng(cfg.seed)
    if target_col not in task_columns:
        raise ValueError(f"target_col {target_col!r} not in task_columns {task_columns}")
    target_idx = task_columns.index(target_col)

    train_df = train_df.copy()
    n = len(train_df)
    val_n = max(1, int(np.ceil(cfg.val_fraction * n)))
    perm = rng.permutation(n)
    val_df = train_df.iloc[perm[:val_n]].reset_index(drop=True)
    tr_df = train_df.iloc[perm[val_n:]].reset_index(drop=True)

    train_ds, val_ds, test_ds = _build_datasets(
        tr_df, val_df, test_df, cfg.smiles_col, task_columns, cp_data, featurizers
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=cp_data.collate_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=cp_data.collate_batch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=cp_data.collate_batch,
    )

    # Standardize per-task on training labels (chemprop helper handles masked NaNs).
    # `normalize_targets` returns an sklearn StandardScaler; the FFN's `output_transform`
    # must be a chemprop nn.Module (UnscaleTransform), which Lightning .train()/.eval()s.
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    if cfg.pretrained_checkpoint:
        mpnn = models.MPNN.load_from_checkpoint(cfg.pretrained_checkpoint)
        if cfg.freeze_mpnn:
            for p in mpnn.message_passing.parameters():
                p.requires_grad = False
        # Replace head to match the current task count.
        ffn = nn.RegressionFFN(
            n_tasks=len(task_columns),
            input_dim=mpnn.message_passing.output_dim,
            hidden_dim=cfg.hidden_size,
            n_layers=cfg.ffn_num_layers,
            dropout=cfg.dropout,
            output_transform=output_transform,
        )
        mpnn.predictor = ffn
    else:
        mp = nn.BondMessagePassing(depth=cfg.depth, d_h=cfg.hidden_size, dropout=cfg.dropout)
        agg = nn.MeanAggregation()
        ffn = nn.RegressionFFN(
            n_tasks=len(task_columns),
            input_dim=cfg.hidden_size,
            hidden_dim=cfg.hidden_size,
            n_layers=cfg.ffn_num_layers,
            dropout=cfg.dropout,
            output_transform=output_transform,
        )
        mpnn = models.MPNN(
            message_passing=mp, agg=agg, predictor=ffn,
            batch_norm=True, metrics=[nn.metrics.MAE()],
            init_lr=cfg.init_lr, max_lr=cfg.max_lr, final_lr=cfg.final_lr,
        )

    pl.seed_everything(cfg.seed, workers=True)
    # Force a single-process cluster environment. On Cray/Slurm (e.g. LUMI), an interactive
    # `srun --pty bash` allocation exports PMI_* env vars; Lightning's auto-detection skips
    # SLURMEnvironment for interactive shells and falls through to MPIEnvironment, which inits
    # cray-mpich PMI and aborts ("PMI_Init returned -1") because this nested worker isn't its
    # own srun step. We always run single-node/single-device, so pin LightningEnvironment.
    from lightning.pytorch.plugins.environments import LightningEnvironment
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        num_nodes=1,
        plugins=[LightningEnvironment()],
        enable_progress_bar=False,
        logger=False,
        # No automatic checkpointing: many array tasks share this working dir, and
        # Lightning's default ModelCheckpoint auto-versions filenames in a shared
        # `checkpoints/` dir, which races under high concurrency (FileNotFoundError on
        # `...-vN.ckpt`). We predict in-process right after fit; explicit persistence
        # is handled separately via `checkpoint_out` below.
        enable_checkpointing=False,
        gradient_clip_val=cfg.grad_clip,
        deterministic=True,
    )
    trainer.fit(mpnn, train_loader, val_loader)

    if checkpoint_out is not None:
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_out))

    preds = trainer.predict(mpnn, test_loader)
    Y_hat = np.concatenate([p.cpu().numpy() for p in preds], axis=0)
    return Y_hat[:, target_idx]
