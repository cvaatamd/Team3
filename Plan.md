# Multi-task few-shot transfer learning for ADMET — build plan

A spec for an agentic system that demonstrates **when external and cross-task data lift few-shot
ADMET prediction**, evaluated on the OpenADMET–ExpansionRx challenge test split.

This document is the brief for Claude Code. Build the deterministic toolset first and get one
target endpoint working end-to-end, then wrap the three agents around it. Do **not** start with the
agent graph.

---

## 0. Thesis and what "done" looks like

**Thesis.** For a sparse target ADMET endpoint, training data from (a) richer ExpansionRx endpoints
(cross-task) and (b) external ADMET datasets (cross-dataset) lowers error most in the low-data
regime, and the size of that lift depends on how well the auxiliary assay matches the target. We
demonstrate this with subsampled learning curves and explain the pattern.

**Primary deliverables**
1. `n`-vs-RAE learning curves per endpoint, with error bands over seeds, for four arms (defined in §4).
2. An aggregate `n`-vs-MA-RAE curve (the challenge's headline metric).
3. A written characterization — produced by the planner agent — of *which* auxiliary tasks transfer,
   *in what data regime*, and *by how much*.
4. A reproducible agentic pipeline (planner / data / ML) that produced all of the above.

**Definition of done for the MVP.** Two target endpoints fully run through the sweep:
`HLM CLint` (strong external match) and one of `MBPB`/`Caco-2 Papp A>B` (weak/absent external match),
showing the contrast. Everything else is upside.

---

## 1. Scoring — match this exactly

- **Metric: MA-RAE** (macro-averaged relative absolute error). Per endpoint, the MAE is normalized by
  that endpoint's dynamic range on the test set, then averaged across endpoints. **Verify the exact
  RAE definition against the official scoring code** in the challenge HuggingFace Space
  (`openadmet/OpenADMET-ExpansionRx-Challenge`) before trusting any number — "relative absolute error"
  is ambiguous (range-normalized MAE vs. error relative to a mean predictor). Implement it as a
  pluggable metric so the definition can be swapped.
- **Split: official time-split.** Use the released splits directly; do not re-split.
  `train` = 5.33k molecules, `test` = 2.28k molecules, both now labeled (challenge closed 2026-01-19).
- Keep MAE / RMSE / R² / Spearman alongside RAE as diagnostics, but RAE/MA-RAE is the curve axis.

**Eval hygiene (non-negotiable).** The competition `test` split is touched exactly once, at evaluation.
Subsampling, model selection, and any hyperparameter tuning happen on the `train` split only
(carve an internal validation fold out of `train`). Never let test data influence a training pool.

---

## 2. Data

### 2.1 ExpansionRx (target dataset)

Source: `openadmet/openadmet-expansionrx-challenge-data` on HuggingFace (CC-BY-4.0).

```python
from datasets import load_dataset
train_df = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="train").to_pandas()
test_df  = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="test").to_pandas()
```

Use the `default` (ML-ready, in-range only) subset, not `raw` (which carries `>`/`<` censored
modifiers). The data is sparse: each molecule has labels for only a subset of endpoints.

**Columns:** `Molecule Name`, `SMILES`, then the 9 endpoint columns.

| Column | Endpoint | Units | Suggested transform |
|---|---|---|---|
| `LogD` | Lipophilicity | log unit | none (already log) |
| `KSOL` | Kinetic solubility | µM | log10 |
| `HLM CLint` | Human liver microsomal clearance | mL/min/kg | log10 |
| `MLM CLint` | Mouse liver microsomal clearance | mL/min/kg | log10 |
| `Caco-2 Permeability Papp A>B` | Passive permeability | 1e-6 cm/s | log10 |
| `Caco-2 Permeability Efflux` | Efflux ratio | ratio | log10 |
| `MPPB` | Mouse plasma protein binding | % unbound | logit(fu), fu=%/100 |
| `MBPB` | Mouse brain protein binding | % unbound | logit(fu) |
| `MGMB` | Mouse gastrocnemius muscle binding | % unbound | logit(fu) |

(The `raw` subset additionally has Rat Liver Microsomal `RLM CLint` (mL/min/kg), not part of the
official 9 — useful as an extra cross-task auxiliary.)

Train models in transformed space; **invert before computing RAE** so the metric is on the
challenge's native scale. Clip fraction-unbound to e.g. [0.001, 0.999] before logit. Handle zeros/
near-zeros in clearance/solubility with a small floor before log10.

**Coverage is the experiment.** Profile per-endpoint label counts on load. Expect `HLM/MLM CLint` and
`KSOL` to be the richest (natural *source* tasks) and `MBPB`/`MGMB`/`Caco-2` to be sparse (natural
*target* tasks). The data agent uses this profile to pick source/target pairings.

### 2.2 Known data-quality traps (from the challenge post-mortem)

Bake these into the data agent as flags:
- **Compound ID is a time proxy** (`E-00xxxxx` increasing ≈ later in the campaign). Use it to make
  subsampling respect temporal order (sample "earlier" compounds as history) and to reason about the
  time-split.
- **Solubility distribution shift in the first ~15% of compounds** (an assay-concentration artifact).
  Top teams improved by removing that slice. Make exclusion a toggle and test its effect.
- **HLM clearance has a large gap toward the end of the training set** → uneven availability across
  time-based splits. Be aware when subsampling HLM.

### 2.3 External sources (registry for the data agent)

The data agent draws from a **curated registry**, not the open web (provenance/licensing). Endpoint
mapping onto ExpansionRx targets, best first:

| ExpansionRx target | External source | Access | Match quality |
|---|---|---|---|
| `HLM CLint` (mL/min/kg) | Biogen/Fang `LOG HLM_CLint (mL/min/kg)` | Polaris `biogen/adme-fang-v1` | Near-identical (same assay, same units, already log) |
| `RLM CLint` (raw) | Biogen/Fang `LOG RLM_CLint (mL/min/kg)` | Polaris `biogen/adme-fang-v1` | Strong |
| `Caco-2 Efflux` | Biogen/Fang `LOG MDR1-MDCK ER` | Polaris `biogen/adme-fang-v1` | Analogous (different cell line) |
| `KSOL` (µM) | Biogen/Fang `LOG SOLUBILITY PH 6.8 (ug/mL)` | Polaris `biogen/adme-fang-v1` | Needs µg/mL→µM (MW) + protocol caveat |
| `MPPB` (mouse) | Biogen/Fang `hPPB` / `rPPB` (% unbound) | Polaris `biogen/adme-fang-v1` | Species transfer (human/rat→mouse) |
| `LogD` | ChEMBL / TDC lipophilicity | TDC / ChEMBL | Available, noisy/aggregated (stretch) |
| `Caco-2 Papp A>B` | TDC `Caco2_Wang` | `PyTDC` | Different system |
| `MBPB`, `MGMB` | — none public — | — | **Intra-dataset transfer only** |

Secondary source to wire in if time allows: **ASAP–Polaris–OpenADMET antiviral ADMET** (Polaris),
another sparse lead-op ADMET set.

Biogen/Fang: ~3,521 compounds, 6 endpoints (human/rat liver microsomal stability, MDR1-MDCK efflux
ratio, solubility, human/rat plasma protein binding). Already in log space.

```python
# verify current Polaris client API; login may be required for some datasets
import polaris as po
ds = po.load_dataset("biogen/adme-fang-v1")
```

```python
from tdc.single_pred import ADME
data = ADME(name="Caco2_Wang")
```

### 2.4 Harmonization (the data agent's real job)

Per (target, source) pair the data agent must:
1. **Unit reconciliation** — convert source to the target's transformed space. HLM: both → log10
   mL/min/kg (trivial). KSOL: convert Biogen µg/mL → µM via MW (`MW = Descriptors.MolWt(mol)`,
   `µM = µg/mL ÷ MW × 1000`), then log10.
2. **Calibration check** — if any compounds overlap (canonical SMILES / InChIKey), regress source vs
   target to detect a systematic offset/scale difference between labs.
3. **Merge strategy decision** — default to **co-training the source as a separate auxiliary task/head**
   (robust to inter-lab calibration offsets), *not* pooling into the same target column. Only pool
   into one column if the calibration check shows they're aligned. This decision is logged and is one
   of the genuinely agentic choices.
4. **Standardization** — canonical SMILES, salt strip, neutralize (RDKit / `chembl_structure_pipeline`),
   dedupe, drop the flagged bad slices (§2.2) per toggle.
5. **Pool hygiene (required for Chemprop MT).** After merging intra-task and external sources, **drop
   any row that has no label in any task column** before handing the pool to the ML agent. Sparse
   arms (`baseline`, `external`) otherwise retain thousands of all-NaN ExpansionRx rows; with
   Chemprop's masked multitask loss and `batch_size=64`, many training batches then have zero valid
   labels → loss = 0/0 = NaN → all-NaN predictions (`RAE=nan`, `n_test=0`). Multi-task arms with
   intra-task auxiliaries rarely hit this because their extra heads label nearly every row. See
   [CHEMPROP_STABILITY.md](CHEMPROP_STABILITY.md) for the full diagnosis.

---

## 3. Modelling (ML agent)

Default to **Chemprop v2** multitask; the ML agent has flexibility to choose the transfer mechanism.

> Verify the current Chemprop v2.x API before coding — it's PyTorch-Lightning based and the API has
> changed across minor versions. Key capabilities to use: multitask regression with **masked loss over
> NaN targets** (a molecule contributes only to endpoints it has labels for), and transfer via loading
> a pretrained checkpoint with optional frozen MPNN encoder.

Three mechanisms, selectable per arm:
- **MT co-train** — single Chemprop MPNN, one head per task (the 9 ExpansionRx endpoints, plus any
  harmonized external endpoints as extra heads). Masked loss. This is the workhorse and yields the
  multi-target predictions the challenge scores.
- **Pretrain → fine-tune** — pretrain the MPNN on the data-rich source (e.g. Biogen HLM), then
  fine-tune on the `n` target samples; option to freeze the encoder and fit only the head (strongest at
  the smallest `n`).
- **Frozen-embedding + light head** — extract Chemprop (or fingerprint) embeddings, fit ridge/GBM head
  per task. Cheap to run across the whole sweep; good few-shot baseline.

**Mandatory strong baseline: LightGBM** (or XGBoost) on `ECFP4 (2048-bit) + RDKit descriptors`,
single-task. ADMET benchmarks routinely show tuned GBMs beating fancier models — the transfer story
is only credible if the MT/transfer arms beat this. Make it a real, tuned baseline, not a strawman.

All models: train in transformed target space, invert predictions before scoring. Fixed seeds,
deterministic where possible.

**Chemprop MT on LUMI / SLURM (stability requirements).** The wrapper in `src/models/chemprop_mt.py`
must:

- Use **masked loss over NaN targets** (Chemprop built-in) *and* ensure the data agent never feeds
  all-unlabeled rows (§2.4 step 5).
- Set **`enable_checkpointing=False`** when many array tasks share a working directory — Lightning's
  default checkpoint versioning races under concurrency.
- Wire the FFN **`output_transform`** as chemprop's `UnscaleTransform`, not a raw sklearn scaler.
- Pin **`LightningEnvironment`** on Cray/Shasta login allocations to avoid PMI init failures.
- Optionally lower **`max_lr`** and enable **`gradient_clip_val`** on very sparse multitask pools.

Details and debugging checklist: [CHEMPROP_STABILITY.md](CHEMPROP_STABILITY.md).

---

## 4. Experiment protocol (planner-driven)

For each target endpoint, with the official `test` split fixed, sweep training size
`n ∈ {25, 50, 100, 250, 500, full}` (cap at available labels) and run four arms:

1. **Baseline** — single-task, `n` target samples only.
2. **+ intra-task** — MT trunk on `n` target samples + all available ExpansionRx source endpoints.
3. **+ external** — `n` target samples + the harmonized external source for that target.
4. **+ both** — full stack.

Per (endpoint, arm, n): **≥5 seeds** (different subsamples + init), report mean RAE with error band.
A "lift" inside the band is not a lift. Expected signature: transfer arms below baseline at small `n`,
converging as `n` grows — that shape *is* the answer to "when does it help."

Subsampling: respect temporal order via compound ID where it matters; keep the held-out test fixed
across all arms and seeds.

**Outputs per run:** tidy results table (`endpoint, arm, n, seed, rae, mae, r2, spearman`), the curve
plots, and a run manifest (configs, data versions, git SHA).

---

## 5. Agent architecture and contracts

Three agents over a deterministic toolset. The agents are **LLM-driven controllers that call typed
tools**; the heavy lifting (train, eval, harmonize) is deterministic Python. Build the tools first,
then the agents.

```
Planner agent ── dispatches ──▶ Data agent ──(training pool)──▶ ML agent ──(RAE)──▶ results
     ▲                                                                                  │
     └──────────────────────── reads results, picks next arm ───────────────────────────┘
```

Use plain dataclasses (or pydantic) for the contracts so an agent can be a deterministic function
*or* an LLM controller behind the same interface.

### 5.1 Data agent

**Responsibility:** given a target endpoint and requested pool size, return a curated, harmonized
training pool. Decides source selection, unit reconciliation, merge strategy, bad-slice exclusion.

```python
@dataclass
class PoolRequest:
    target_endpoint: str            # e.g. "HLM CLint"
    n: int | None                   # target-label budget; None = all
    include_intra_task: bool        # add ExpansionRx source endpoints
    include_external: bool          # add harmonized external source(s)
    seed: int
    exclude_flagged_slices: bool = True

@dataclass
class TrainingPool:
    df: "pd.DataFrame"              # SMILES + one column per (transformed) task
    task_columns: list[str]         # target first, then auxiliaries
    target_endpoint: str
    provenance: dict                # source ids, versions, n per task, transforms applied
    harmonization_log: dict         # unit conversions, calibration checks, merge decisions
```

### 5.2 ML agent

**Responsibility:** given a training pool and the fixed test set, choose a mechanism, train, evaluate,
return metrics. Has flexibility over mechanism/featurizer/hyperparameters.

```python
@dataclass
class TrainRequest:
    pool: TrainingPool
    mechanism: Literal["mt_cotrain", "pretrain_finetune", "frozen_embed", "gbm_baseline"]
    test_df: "pd.DataFrame"         # official test split, target labels held for scoring only
    seed: int

@dataclass
class EvalResult:
    target_endpoint: str
    arm: str
    n: int
    seed: int
    rae: float                      # on native scale, per challenge definition
    extra_metrics: dict             # mae, rmse, r2, spearman
    model_manifest: dict            # mechanism, hyperparams, checkpoint path
```

### 5.3 Planner agent

**Responsibility:** owns the sweep. Picks target/source pairings from the coverage profile, enumerates
(arm, n, seed) jobs, dispatches to data+ML agents, collects `EvalResult`s, and writes the
characterization report. The "explain when transfer helps" deliverable is the planner's output, not an
afterthought.

```python
@dataclass
class ExperimentPlan:
    target_endpoints: list[str]
    arms: list[str]
    n_grid: list[int | None]
    seeds: list[int]

# planner.run(plan) -> ResultsTable + curves + report.md
```

Agentic decisions worth making real (so the agent layer isn't theater): data agent choosing/ skipping a
noisy source and choosing merge strategy from the calibration check; ML agent choosing mechanism from
pool shape (tiny `n` → frozen-embed/freeze-encoder); planner choosing which pairings to explore next
based on observed lift and writing the interpretation.

---

## 6. Repo layout

```
.
├── PLAN.md
├── pyproject.toml
├── README.md
├── conf/                       # hydra/yaml configs: endpoints, n_grid, seeds, sources
├── src/
│   ├── data/
│   │   ├── expansionrx.py       # load, profile coverage, transforms, slice flags
│   │   ├── registry.py          # external source registry + endpoint mapping
│   │   ├── harmonize.py         # unit reconciliation, calibration, merge strategy
│   │   └── standardize.py       # SMILES canonicalize/strip/neutralize/dedupe
│   ├── models/
│   │   ├── chemprop_mt.py        # MT co-train + pretrain/finetune wrappers
│   │   ├── frozen_embed.py
│   │   └── gbm_baseline.py
│   ├── eval/
│   │   ├── metrics.py            # RAE/MA-RAE (pluggable) + diagnostics
│   │   └── curves.py             # learning-curve plotting
│   ├── agents/
│   │   ├── contracts.py          # dataclasses above
│   │   ├── data_agent.py
│   │   ├── ml_agent.py
│   │   └── planner.py
│   └── experiment/
│       └── runner.py             # job enumeration, seeding, manifests
├── tests/
└── results/                    # tidy tables, plots, run manifests, report.md
```

---

## 7. Build sequence (milestones)

**M0 — scaffold + data.** Repo, env, load ExpansionRx, coverage profile, transforms, standardization,
slice flags. Test: counts and ranges match §2.1.

**M1 — metric + GBM baseline + first curve.** Implement RAE/MA-RAE (verified against official scoring).
LightGBM single-task on `HLM CLint`. Produce the baseline `n`-vs-RAE curve with seeds. *This is the
end-to-end skeleton — everything hangs off it.*

**M2 — Chemprop MT + intra-task arm.** MT co-train across ExpansionRx endpoints; masked NaN loss.
Add the `+ intra-task` arm for HLM. Confirm it beats (or doesn't) the GBM honestly.

**M3 — external transfer for HLM.** Wire Polaris `biogen/adme-fang-v1`, harmonize HLM (log10 mL/min/kg),
calibration check, co-train as auxiliary head. Add `+ external` and `+ both` arms. **First complete
transfer story.**

**M4 — contrast endpoint.** Repeat M2–M3 for `MBPB` (no external source → intra-task only) or
`Caco-2 Papp A>B` (TDC `Caco2_Wang`). Show the contrast in match quality.

**M5 — wrap agents.** Put the data/ML/planner controllers around the deterministic tools using the
§5 contracts. Planner runs the full sweep and emits `report.md`.

**M6 — stretch.** ASAP antiviral source; pretrain/finetune + frozen-embed mechanisms; all 9 endpoints;
LLM-in-the-loop decisions for the agents; submit to the (now-open) leaderboard for an external check.

---

## 8. First-target spec — HLM CLint (do this first, concretely)

- **Target column:** `HLM CLint` (mL/min/kg) → `log10`. Floor zeros/near-zeros at a small value first.
- **Intra-task auxiliaries:** all other ExpansionRx endpoints (`MLM CLint` especially — closely related
  rodent analog), transformed per §2.1, as extra heads with masked loss.
- **External source:** Biogen/Fang `LOG HLM_CLint (mL/min/kg)` from `biogen/adme-fang-v1`. Already
  log10 mL/min/kg → directly comparable.
  - Calibration check: InChIKey overlap with ExpansionRx; regress to detect lab offset.
  - Merge: default **auxiliary task/head** (separate Biogen-HLM head sharing the MPNN trunk), not pooled
    into the ExpansionRx HLM column, unless calibration says aligned.
- **Arms:** baseline (LightGBM + single-task Chemprop), + intra-task, + external, + both.
- **Sweep:** `n ∈ {25,50,100,250,500,full}` of ExpansionRx HLM labels, 5 seeds.
- **Expect:** external arm gives the cleanest lift at small `n` (near-identical assay); intra-task adds
  from MLM. Quantify the gap and where it closes.

---

## 9. Guardrails

- Test split touched once, at scoring, ever. Tune on an internal `train` fold only.
- Report error bands; never claim a lift inside the noise.
- Beat the tuned GBM or say so plainly.
- Log provenance + harmonization decisions for every pool (reproducibility + the agentic-decision
  narrative).
- Verify before trusting: the RAE definition (official scoring), the Chemprop v2 API, and the Polaris
  client API are the three things most likely to have drifted — check current docs/source.
- External data is co-trained, not blindly pooled, unless a calibration check justifies pooling.

---

## 10. Environment

```
python >= 3.10
chemprop >= 2.0            # verify current API
rdkit
lightgbm                  # or xgboost
scikit-learn
datasets                  # HuggingFace, for ExpansionRx
polaris-lib               # Polaris hub (verify package/import name)
PyTDC                     # TDC external sources
pandas, numpy, scipy
matplotlib                # curves
hydra-core / pydantic     # configs + contracts
pytest
```