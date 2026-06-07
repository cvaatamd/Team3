# Option A — side-by-side: original `EXPERIMENT.md` vs the proposed additions

The production report [`../results-fullexp/EXPERIMENT.md`](../results-fullexp/EXPERIMENT.md) is left
**unchanged**. This file shows, at each of the three places Option A would touch, the **current text
(left)** next to the **Option-A enhanced text (right)**. Option A is the no-compute rigor/framing
pass: (1) a magnitude-vs-ranking headline, (2) a paired seed-level bootstrap significance table, and
(3) upgrading the "official RAE" note to "verified". Everything else in the report is identical.

All Option-A numbers come from the existing `results-fullexp/*/results.parquet` and the already-run
[`outputs/seed_bootstrap.txt`](./outputs/seed_bootstrap.txt) — no new training.

---

## Change 1 — add a headline at the top of `## Results`

<table>
<tr>
<th width="50%">results-fullexp/EXPERIMENT.md (current)</th>
<th width="50%">+ Option A</th>
</tr>
<tr valign="top">
<td>

<p><b>## Results</b></p>
<p>All 600 jobs completed (no failures). Tables show <b>mean RAE ± std over 5 seeds</b> (lower is
better); <b>bold</b> = best arm at that data size. Arms: <code>+intra</code> = ExpansionRx multitask,
<code>+external</code> = Biogen multitask, <code>+both</code> = both auxiliary sources.</p>
<p><i>(per-endpoint tables follow…)</i></p>

</td>
<td>

<p><b>## Results</b></p>
<p><b>### Headline: external data helps <i>magnitude</i>, intra-task helps <i>ranking</i></b></p>
<p>The central result is a <b>metric-dependent split</b>: whether external (cross-dataset) assays
help depends on <i>what you measure</i>. At full data, on the three endpoints with transfer headroom:</p>

<table>
<tr><th>Endpoint</th><th>Best on <b>magnitude</b> (RAE/MAE/RMSE/R²)</th><th>Best on <b>ranking</b> (Spearman)</th></tr>
<tr><td>MBPB</td><td><b>external</b> (RAE 0.096 vs 0.121; R² 0.43 vs 0.11)</td><td>both (ρ 0.834) — external 0.822</td></tr>
<tr><td>KSOL</td><td><b>external</b> (RAE 0.270 vs 0.329; RMSE 116 vs 137)</td><td><b>intra</b> (ρ 0.544) — external 0.515</td></tr>
<tr><td>MPPB</td><td><b>external</b> (RAE 0.118 vs 0.161; R² 0.38 vs −0.12)</td><td><b>intra</b> (ρ 0.738) — external 0.723</td></tr>
</table>

<p><b>External maximises calibrated accuracy on every error metric (3/5 endpoints, up to −27% RAE),
yet wins ranking on none</b> — there, intra-task co-training (or <code>+both</code>) leads. A
ranking-only evaluation concludes "external adds nothing" (true on ranking), but that does not
generalise to calibrated prediction (dose projection, PK, go/no-go thresholds), where the harmonised
external source is the single best auxiliary in 3/5 endpoints.</p>

<blockquote><b>Decision rule:</b> external for magnitude, intra-task for ranking. Reading one metric
alone hides half the story (Caco-2 Efflux: tiny RAE but negative R² — range-saturated).</blockquote>

<hr>
<p>All 600 jobs completed (no failures). Tables show <b>mean RAE ± std over 5 seeds</b> …
<i>(unchanged intro + per-endpoint tables follow)</i></p>

</td>
</tr>
</table>

---

## Change 2 — add a significance subsection after the cross-round synthesis

<table>
<tr>
<th width="50%">results-fullexp/EXPERIMENT.md (current)</th>
<th width="50%">+ Option A</th>
</tr>
<tr valign="top">
<td>

<p><b>Decision rule the agent learned:</b> spend effort on transfer when the single-task baseline is
weak; prefer the harmonized external assay at moderate-to-full data and add intra-task co-training in
the few-shot regime; skip transfer when the baseline is already strong or near-saturated.</p>

<p><b>## Alternative metrics (not just RAE)</b></p>
<p><i>(…)</i></p>

</td>
<td>

<p><b>Decision rule the agent learned:</b> … <i>(unchanged)</i></p>

<p><b>### Statistical significance (paired seed-level bootstrap)</b></p>
<p>The transfer lift over the baseline at full data is <b>not</b> seed noise. Paired bootstrap (10k
resamples over 5 seeds; Δ = baseline − transfer, &gt;0 ⇒ transfer better; CI excludes 0 ⇒ significant):</p>

<table>
<tr><th>endpoint</th><th>arm</th><th>metric</th><th>Δ (lift)</th><th>95% CI</th><th>signs</th></tr>
<tr><td>MBPB</td><td>external</td><td>RAE</td><td>+0.0253</td><td>[+0.0165, +0.0352]</td><td>5/5</td></tr>
<tr><td>KSOL</td><td>external</td><td>RAE</td><td>+0.0593</td><td>[+0.0439, +0.0747]</td><td>5/5</td></tr>
<tr><td>MPPB</td><td>external</td><td>RAE</td><td>+0.0434</td><td>[+0.0335, +0.0522]</td><td>5/5</td></tr>
<tr><td>MBPB</td><td>external</td><td>Spearman</td><td>+0.0734</td><td>[+0.0440, +0.1056]</td><td>5/5</td></tr>
<tr><td>KSOL</td><td>external</td><td>Spearman</td><td>+0.0303</td><td>[+0.0155, +0.0451]</td><td>5/5</td></tr>
<tr><td>MPPB</td><td>external</td><td>Spearman</td><td>+0.2357</td><td>[+0.1805, +0.2909]</td><td>5/5</td></tr>
<tr><td>MBPB</td><td>both</td><td>RAE</td><td>+0.0179</td><td>[+0.0044, +0.0358]</td><td>4/5</td></tr>
<tr><td>MPPB</td><td>both</td><td>RAE</td><td>+0.0322</td><td>[+0.0245, +0.0386]</td><td>5/5</td></tr>
</table>

<p>Every external-arm lift on MBPB/KSOL/MPPB is <b>significant on both RAE and Spearman</b>, same sign
in all 5 seeds. <i>Caveat:</i> bootstraps over 5 seeds (seed variance, not test-molecule variance);
<code>further-improvements/</code> persists per-molecule preds for molecule-level CIs. Reproduce:
<code>python further-improvements/bootstrap_lift.py</code>.</p>

<p><b>## Alternative metrics (not just RAE)</b></p>

</td>
</tr>
</table>

---

## Change 3 — upgrade the "official RAE" note in *Alternative metrics*

<table>
<tr>
<th width="50%">results-fullexp/EXPERIMENT.md (current)</th>
<th width="50%">+ Option A</th>
</tr>
<tr valign="top">
<td>

<p>A different <i>RAE definition</i> is also pluggable (<code>range_normalized</code> (default),
<code>vs_mean</code>, or the <code>official</code> <b>stub</b>) via <code>rae_definition</code> in
<code>src/eval/metrics.py</code>; changing it recomputes from the stored predictions.</p>

</td>
<td>

<p>A different <i>RAE definition</i> is also pluggable (<code>range_normalized</code> (default),
<code>vs_mean</code>, or <code>official</code>) via <code>rae_definition</code> in
<code>src/eval/metrics.py</code>. The <code>official</code> branch is now implemented and
<b>verified ≡ <code>range_normalized</code></b> (max abs diff <code>0.00e+00</code> over all cells,
re-scored offline from persisted predictions — see <code>../further-improvements/</code>), so the RAE
percentages above are the challenge's headline metric, not an internal proxy.</p>

</td>
</tr>
</table>

---

## How to apply Option A (if you decide to)

These three edits go into `../results-fullexp/EXPERIMENT.md` (the right-hand column in each table
above). They are additive and reversible. The source material lives here:

- Headline (Change 1): [`metric_reconciliation.md`](./metric_reconciliation.md)
- Significance table (Change 2): [`outputs/seed_bootstrap.txt`](./outputs/seed_bootstrap.txt)
  (regenerate with `python further-improvements/bootstrap_lift.py`)
- Official-RAE verification (Change 3): `python further-improvements/scripts/rescore_official.py`
