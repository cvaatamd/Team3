"""Load and transform the OpenADMET-ExpansionRx challenge dataset.

§2.1 of Plan.md is the source of truth for the endpoint table and transforms.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

def _resolve_endpoint_conf() -> Path:
    """Locate conf/endpoints.yaml robustly.

    The patched copy lives under further-improvements/code/, one level deeper than the original
    src/, so the old parents[2] guess misses the repo's conf/. Prefer (1) $ADMET_CONF_DIR,
    (2) <cwd>/conf (drivers run with --pwd at the repo root), then (3) a few parents up.
    """
    import os
    env = os.environ.get("ADMET_CONF_DIR")
    candidates = []
    if env:
        candidates.append(Path(env) / "endpoints.yaml")
    candidates.append(Path.cwd() / "conf" / "endpoints.yaml")
    here = Path(__file__).resolve()
    candidates += [p / "conf" / "endpoints.yaml" for p in here.parents]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd() / "conf" / "endpoints.yaml"


ENDPOINT_CONF = _resolve_endpoint_conf()
HF_DATASET_ID = "openadmet/openadmet-expansionrx-challenge-data"
SMILES_COL = "SMILES"
ID_COL = "Molecule Name"

# Slice exclusion flags — see §2.2.
EARLY_KSOL_FRACTION = 0.15  # solubility-distribution shift in first ~15% of compounds


@dataclass(frozen=True)
class EndpointSpec:
    column: str
    transform: str
    floor: float | None = None
    clip: tuple[float, float] | None = None

    @property
    def transformed_column(self) -> str:
        return f"{self.column}__t"


@dataclass
class ExpansionRxData:
    train: pd.DataFrame
    test: pd.DataFrame
    endpoints: list[EndpointSpec]
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)


def _load_endpoints(path: Path = ENDPOINT_CONF) -> list[EndpointSpec]:
    raw = yaml.safe_load(path.read_text())["endpoints"]
    specs = []
    for entry in raw:
        clip = tuple(entry["clip"]) if "clip" in entry else None
        specs.append(
            EndpointSpec(
                column=entry["column"],
                transform=entry["transform"],
                floor=entry.get("floor"),
                clip=clip,
            )
        )
    return specs


def load_expansionrx(subset: str = "default") -> ExpansionRxData:
    """Load both splits as pandas. `default` = ML-ready, in-range only (§2.1)."""
    from datasets import load_dataset  # local import: heavy dep

    train = load_dataset(HF_DATASET_ID, subset, split="train").to_pandas()
    test = load_dataset(HF_DATASET_ID, subset, split="test").to_pandas()
    endpoints = _load_endpoints()
    apply_transforms(train, endpoints)
    apply_transforms(test, endpoints)
    cov = profile_coverage(train, endpoints)
    log.info("ExpansionRx loaded: train=%d, test=%d", len(train), len(test))
    return ExpansionRxData(train=train, test=test, endpoints=endpoints, coverage=cov)


def apply_transforms(df: pd.DataFrame, endpoints: Iterable[EndpointSpec]) -> None:
    """Add transformed columns in-place. Inverse is `invert_transform`."""
    for ep in endpoints:
        if ep.column not in df.columns:
            log.warning("endpoint column %r missing from frame", ep.column)
            continue
        x = df[ep.column].astype(float)
        df[ep.transformed_column] = _forward(x, ep)


def _forward(x: pd.Series | np.ndarray, ep: EndpointSpec) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if ep.transform == "identity":
        return x
    if ep.transform == "log10":
        floor = ep.floor or 1e-9
        return np.log10(np.maximum(x, floor))
    if ep.transform == "logit_pct":
        # Values are % unbound; fu = pct/100.
        fu = x / 100.0
        lo, hi = ep.clip or (1e-3, 1 - 1e-3)
        fu = np.clip(fu, lo, hi)
        return np.log(fu / (1.0 - fu))
    raise ValueError(f"unknown transform: {ep.transform}")


def invert_transform(y: np.ndarray, ep: EndpointSpec) -> np.ndarray:
    """Map predictions back to challenge-native units before scoring (§2.1)."""
    y = np.asarray(y, dtype=float)
    if ep.transform == "identity":
        return y
    if ep.transform == "log10":
        return np.power(10.0, y)
    if ep.transform == "logit_pct":
        fu = 1.0 / (1.0 + np.exp(-y))
        return fu * 100.0
    raise ValueError(f"unknown transform: {ep.transform}")


def profile_coverage(df: pd.DataFrame, endpoints: Iterable[EndpointSpec]) -> pd.DataFrame:
    """Per-endpoint label counts. The data agent uses this to pick source/target pairs."""
    rows = []
    n = len(df)
    for ep in endpoints:
        if ep.column not in df.columns:
            rows.append({"endpoint": ep.column, "n_labeled": 0, "frac": 0.0})
            continue
        n_lab = int(df[ep.column].notna().sum())
        rows.append({"endpoint": ep.column, "n_labeled": n_lab, "frac": n_lab / max(n, 1)})
    return pd.DataFrame(rows).sort_values("n_labeled", ascending=False).reset_index(drop=True)


def temporal_order_index(df: pd.DataFrame) -> pd.Series:
    """Order compounds by ID — E-00xxxxx increases ≈ later in campaign (§2.2)."""
    s = df[ID_COL].astype(str).str.extract(r"E-?(\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)


def flagged_slice_mask(df: pd.DataFrame, endpoint: str) -> pd.Series:
    """Return a boolean mask of rows to *exclude* for the given endpoint (§2.2)."""
    mask = pd.Series(False, index=df.index)
    if endpoint == "KSOL":
        order = temporal_order_index(df)
        order_rank = order.rank(pct=True)
        # drop the earliest 15% — solubility assay-concentration artifact
        mask |= order_rank <= EARLY_KSOL_FRACTION
    return mask


def subsample_target(
    df: pd.DataFrame,
    endpoint: str,
    n: int,
    seed: int,
    *,
    respect_temporal_order: bool = False,
    exclude_flagged: bool = True,
) -> pd.DataFrame:
    """Carve `n` labels for `endpoint` from `df` for use as the training pool target column.

    Rows missing the target label are kept (they may carry auxiliary labels useful in MT arms)
    — only the *budget* on the target itself is capped at `n`. We return the full frame with the
    target column masked outside the sampled n rows so the masked-loss MT setup works as expected.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    if exclude_flagged:
        bad = flagged_slice_mask(df, endpoint)
        df = df.loc[~bad].copy()

    have = df.index[df[endpoint].notna()]
    if n >= len(have):
        return df

    if respect_temporal_order:
        order = temporal_order_index(df.loc[have])
        # Sample with probability biased toward earlier compounds.
        ranks = order.rank(pct=True).values
        weights = (1.0 - ranks) + 1e-3
        probs = weights / weights.sum()
        chosen = rng.choice(have, size=n, replace=False, p=probs)
    else:
        chosen = rng.choice(have, size=n, replace=False)
    drop = have.difference(pd.Index(chosen))
    df.loc[drop, endpoint] = np.nan
    # rebuild transformed col since we mutated raw
    spec = next((e for e in _load_endpoints() if e.column == endpoint), None)
    if spec is not None:
        df[spec.transformed_column] = _forward(df[endpoint], spec)
    return df
