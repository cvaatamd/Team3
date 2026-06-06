# Multi-task few-shot ADMET — characterization

- Endpoints: KSOL
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 80

## Best arm per (endpoint, n)

### KSOL

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.4136 | 0.4286 |     0.4286 |       0.4286 |
|  50 |     0.424  | 0.3817 |     0.3811 |       0.4056 |
| 100 |     0.4272 | 0.4211 |     0.4023 |       0.4555 |
| 250 |     0.3961 | 0.364  |     0.3392 |       0.375  |


## Interpretation

- **KSOL / intra_task**: largest lift 0.0211 RAE at n=250 (baseline=0.3961, intra_task=0.3750).
- **KSOL / external**: largest lift 0.0570 RAE at n=250 (baseline=0.3961, external=0.3392).
- **KSOL / both**: largest lift 0.0424 RAE at n=50 (baseline=0.4240, both=0.3817).

## LLM characterization

Across the KSOL endpoint, the external auxiliary task consistently lowers RAE relative to the baseline, especially in the high‑data regime (n=250) where its mean RAE drops from 0.396 to 0.339 (‑14.4% relative improvement) and its 95% error band (0.318–0.360) lies entirely below the baseline band (0.371–0.421). The "both" arm also reduces mean RAE at n=250 (0.364 vs 0.396, ‑8.1%) and shows a tighter variance (std = 0.011) but its upper band (0.375) still overlaps the baseline lower bound, so the lift is not statistically confirmed. At n=50 and n=100 the external arm yields modest mean reductions (‑9.9% at n=50, ‑5.8% at n=100) but its error bands overlap the baseline, indicating no clear statistical lift. The intra‑task arm never beats the baseline; its means are equal or higher and its bands always overlap or exceed baseline performance. Overall, transfer benefits are evident only for the external task and only become reliable when the target training set reaches ~250 samples.

### Per-endpoint

**KSOL_n=25** — All transfer arms (both, external, intra_task) have higher mean RAE (≈0.429) than baseline (0.414) and larger std, indicating degradation.

**KSOL_n=50** — Both (0.382 ± 0.027) and external (0.381 ± 0.046) improve mean RAE versus baseline (0.424 ± 0.020) by ~9–10%, but their upper error bounds (0.408 and 0.427) overlap the baseline lower bound (0.404), so the lift is not statistically confirmed. Intra_task (0.406 ± 0.038) is marginally better than baseline mean but still overlaps.

**KSOL_n=100** — External (0.402 ± 0.027) lowers mean RAE by ~5.8% relative to baseline (0.427 ± 0.018), yet its upper bound (0.429) touches the baseline mean, so the improvement is not definitive. Both and intra_task show no clear benefit.

**KSOL_n=250** — External achieves a clear lift: mean RAE 0.339 ± 0.021 (‑14.4% vs baseline 0.396 ± 0.025) with the entire error band (0.318–0.360) below the baseline band (0.371–0.421). Both improves mean RAE to 0.364 ± 0.011 (‑8.1%) but its upper bound (0.375) overlaps baseline lower bound, so statistical significance is uncertain. Intra_task (0.375 ± 0.029) offers no reliable gain.

### Open questions

- What properties of the external auxiliary task (e.g., chemical space overlap, label correlation) drive the strong transfer at larger sample sizes?
- Why does the "both" arm not achieve a statistically significant lift despite lower mean RAE and reduced variance at n=250?
- Can alternative weighting or curriculum strategies between the two auxiliary tasks enhance the combined transfer effect?
- Would increasing the number of repetitions (currently count=5) reduce variance enough to detect modest lifts at n=50–100?
- Is the lack of benefit from intra‑task transfer due to task redundancy or insufficient diversity in the auxiliary labels?