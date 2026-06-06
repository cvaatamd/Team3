# What this workflow is for — goal and outputs

A plain-language companion to [Plan.md](Plan.md) (the technical spec) and
[RESULTS.md](RESULTS.md) (the findings). This explains *why* the pipeline exists and *what you
get* when you run it. Chemprop reliability notes: [CHEMPROP_STABILITY.md](CHEMPROP_STABILITY.md).

## The scientific question

ADMET assays are expensive, so for many endpoints you only have a handful of labeled molecules
(the "few-shot" regime). The thesis (Plan.md §0):

> For a sparse target endpoint, borrowing labels from (a) **related ExpansionRx endpoints**
> (cross-task) and (b) **external ADMET datasets** (cross-dataset) lowers prediction error most
> when target data is scarce — and the size of that benefit depends on how well the auxiliary
> assay matches the target.

The workflow is **not** trying to win a leaderboard. It answers a question: *when does auxiliary
data help few-shot ADMET prediction, in what data regime, and by how much?*

## What the workflow does (the agent loop)

For every combination of `(endpoint, arm, n, seed)`:

```
Planner (plans sweep, dispatches, interprets)
   │
   ▼
Data sources ─► Data agent (finds & harmonizes) ──pool(n)──► ML agent (trains & evaluates)
                                                                  │
                            n-vs-RAE curves  ◄────────────────────┤
                            (baseline vs transfer)                ▼
                                                          Held-out test (official MA-RAE)
```

- **Planner** — enumerates the sweep (which endpoints, arms, training sizes `n`, seeds) and, at
  the end, writes the interpretation.
- **Data agent** — builds the training pool: selects related/external sources, reconciles units,
  runs a calibration check, and decides whether to pool a source or co-train it as an auxiliary
  head. For sparse arms (`baseline`, `external`) it also **drops rows with no label in any task
  column** so Chemprop's masked multitask loss never sees empty batches (see
  [CHEMPROP_STABILITY.md](CHEMPROP_STABILITY.md)).
- **ML agent** — picks a mechanism (LightGBM baseline / multitask Chemprop / pretrain-finetune /
  frozen-embed), trains, and scores on the **fixed held-out test split** with the challenge's
  RAE metric.

The **arms** are the controlled comparison (Plan.md §4):

1. `baseline` — target labels only
2. `+ intra_task` — target + other ExpansionRx endpoints
3. `+ external` — target + a harmonized outside dataset
4. `+ both` — full stack

Sweeping `n` (25 → full) across seeds turns this into **learning curves** — error vs. amount of
target data, one line per arm. The *shape* of those curves is the answer: transfer arms dipping
below baseline at small `n` and converging as `n` grows means "transfer helps, but only when
target data is scarce."

## What you get when you run it

Artifacts land in the chosen results dir (e.g. `results/` or `results-agent/`):

| File | What it is |
|---|---|
| `results.parquet` | Tidy table: `endpoint, arm, n, seed, rae, mae, rmse, r2, spearman` for every job |
| `curves.png` | RAE-vs-`n` learning curves per endpoint, with error bands over seeds, baseline vs transfer |
| `ma_rae.png` | The aggregate macro-averaged RAE (the challenge's headline metric) |
| `report.md` | The written characterization — Aitta writes the narrative of which auxiliary tasks transfer, where, and by how much |
| `runs/*.json` | Per-job provenance (sources used, harmonization decisions, mechanism, git SHA) |

## A concrete example of the payoff

The full 240-job sweep in `results/` (see RESULTS.md) already produced this kind of answer:

- **MBPB** (mouse brain binding, no good external match): transfer **clearly helps** —
  `both`/`intra_task` cut RAE ~25–35% vs. baseline once `n ≥ 50`, with non-overlapping error
  bands. Most of the lift comes from the in-distribution ExpansionRx auxiliaries, not the
  external proxy.
- **HLM CLint** (liver clearance, near-identical external assay available): transfer
  **doesn't help** — the plain single-task baseline wins at every `n`.

That contrast *is* the deliverable: borrowing data helps for the sparse, hard-to-source endpoint
and is unnecessary for the data-rich one — a quantified, defensible statement with curves to back
it.

## In one sentence

You get a **reproducible, agent-run experiment** that produces learning curves plus an
LLM-written report telling you, per ADMET endpoint, **whether and when auxiliary/external data is
worth using** — with the agentic layer (Aitta) making the judgment calls (which sources to trust,
how to merge them, which model mechanism, how to interpret the numbers).

When you run several rounds, each round re-exercises the whole plan → data → train → interpret
loop and writes its own curves + report, so you can compare decisions/configs across rounds.

## How to run it

See [Readme.md](Readme.md):

- **Standalone agent, one process** (mini plan): `bash scripts/run_agent.sh`
- **Agent dispatching parallel SLURM jobs** (full sweep): `bash scripts/run_agent.sh --slurm`
- **Connectivity + plan check only**: `bash scripts/run_agent.sh --check`
