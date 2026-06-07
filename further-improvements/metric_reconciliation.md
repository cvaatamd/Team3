# Headline Results section — paste near the top of `results-fullexp/EXPERIMENT.md`

> Ready to merge verbatim. Numbers are pulled from the existing full-data tables in EXPERIMENT.md
> (no new runs). Drop it in just under the `## Results` heading as the headline framing.

---

## Headline: external data helps *magnitude*, intra-task helps *ranking*

The central result of this study is a **metric-dependent split** that resolves an apparent
contradiction in few-shot ADMET transfer: whether external (cross-dataset) assays help at all
depends entirely on *what you measure*.

At full data, across the three endpoints where transfer has headroom (MBPB, KSOL, MPPB):

| Endpoint | Best on **magnitude** (RAE/MAE/RMSE/R²) | Best on **ranking** (Spearman) |
|---|---|---|
| MBPB | **external** (RAE 0.096 vs 0.121; R² 0.43 vs 0.11) | both (ρ 0.834) — external 0.822 |
| KSOL | **external** (RAE 0.270 vs 0.329; RMSE 116 vs 137) | **intra** (ρ 0.544) — external 0.515 |
| MPPB | **external** (RAE 0.118 vs 0.161; R² 0.38 vs −0.12) | **intra** (ρ 0.738) — external 0.723 |

**The external assay maximises calibrated accuracy on every error metric (3/5 endpoints, up to
−27% RAE), yet wins the ranking metric on none of them** — there, intra-task ExpansionRx
co-training (or `+both`) leads. The two objectives genuinely diverge.

**Why this matters.** A ranking-only evaluation (Spearman, or a selection oracle scored on ρ)
concludes "distribution-shifted external assays add nothing" — and on *ranking*, that is correct
here too. But it does **not** generalise to calibrated prediction: for any use-case that needs a
number on the native scale (dose projection, PK modelling, go/no-go thresholds), the harmonised
external source is the single best auxiliary in 3/5 endpoints.

> **Decision rule:** choose the auxiliary source by objective — **external for magnitude,
> intra-task for ranking.** Reading one metric alone hides half the story (see Caco-2 Efflux: tiny
> RAE but negative R² — range-saturated, not error-saturated).

This reconciles the two readings of the same data: the ranking view and the magnitude view are
both right, and the contribution is showing exactly where each applies.
