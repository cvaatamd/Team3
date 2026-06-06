# Multi-task few-shot ADMET — characterization

- Endpoints: HLM CLint
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 120

## Best arm per (endpoint, n)

### HLM CLint

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.105  | 0.1866 |     0.1866 |       0.1866 |
|  50 |     0.1037 | 0.111  |     0.1317 |       0.1104 |
| 100 |     0.1063 | 0.1186 |     0.1065 |       0.1446 |
| 250 |     0.1002 | 0.115  |     0.107  |       0.1207 |
| 500 |     0.0926 | 0.1621 |     0.0979 |       0.1203 |


## Interpretation

- **HLM CLint / intra_task**: largest lift -0.0067 RAE at n=50.0 (baseline=0.1037, intra_task=0.1104).
- **HLM CLint / external**: largest lift -0.0002 RAE at n=100.0 (baseline=0.1063, external=0.1065).
- **HLM CLint / both**: largest lift -0.0072 RAE at n=50.0 (baseline=0.1037, both=0.1110).

## LLM characterization

Across all data regimes for the HLM CLint endpoint, none of the auxiliary-task transfer arms (both, external, intra_task) consistently lowered the Relative Absolute Error (RAE) relative to the baseline. At the smallest training set (n=25) the transfer arms dramatically increased mean RAE (≈0.19 vs 0.105) and exhibited huge error bands (std ≈0.16), indicating severe degradation. For n=50, 250, 500, and the full‑data condition (n=NaN) the transfer arms remained higher than baseline by 0.006–0.08 RAE units, with error bands that either overlapped baseline (external at n=100) or were larger, so no statistically reliable lift was observed. The only regime where a transfer arm approached baseline performance was the external task at n=100, where mean RAE (0.1065) was within 0.0003 of the baseline mean (0.1063) and the stds overlapped, suggesting parity rather than improvement. Overall, auxiliary-task transfer did not provide a measurable benefit for HLM CLint under any examined data regime.

### Per-endpoint

**HLM CLint** — Baseline RAE decreases with more data (0.105±0.004 at n=25 → 0.085±0.002 full). Both‑task transfer is always worse (e.g., 0.187±0.165 at n=25, 0.162±0.063 at full). External‑task transfer matches baseline only at n=100 (0.107±0.006 vs 0.106±0.005) and is slightly higher elsewhere (e.g., 0.098±0.004 vs 0.093±0.002 at full). Intra‑task transfer is never better (e.g., 0.187±0.165 at n=25, 0.134±0.052 at full). No arm achieves a mean RAE below baseline with a non‑overlapping error band.

### Open questions

- Is the external auxiliary task sufficiently related to HLM CLint to enable positive transfer, or is task similarity too low?
- Would larger pre‑training datasets or more sophisticated multi‑task weighting schemes reveal benefits that are masked at these small sample sizes?
- Can model capacity or architecture adjustments (e.g., deeper encoders) better exploit auxiliary signals without overfitting the noisy small‑n regimes?
- Would incorporating additional, chemically relevant auxiliary endpoints (e.g., CYP450 isoforms) improve transfer performance compared to the current tasks?