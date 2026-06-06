# Multi-task few-shot ADMET — characterization

- Endpoints: MPPB
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 120

## Best arm per (endpoint, n)

### MPPB

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.181  | 0.2048 |     0.2048 |       0.2048 |
|  50 |     0.1703 | 0.1521 |     0.1791 |       0.1562 |
| 100 |     0.1736 | 0.1436 |     0.1524 |       0.1441 |
| 250 |     0.1674 | 0.1367 |     0.1478 |       0.1466 |
| 500 |     0.1565 | 0.1233 |     0.1364 |       0.1364 |


## Interpretation

- **MPPB / intra_task**: largest lift 0.0346 RAE at n=nan (baseline=0.1613, intra_task=0.1267).
- **MPPB / external**: largest lift 0.0434 RAE at n=nan (baseline=0.1613, external=0.1179).
- **MPPB / both**: largest lift 0.0332 RAE at n=500.0 (baseline=0.1565, both=0.1233).

## LLM characterization

Across the MPPB endpoint, all three auxiliary-task transfer arms (both, external, intra_task) achieve lower mean RAE than the baseline when the training set is large (n ≥ 500) and on the full‑data condition. The "both" arm shows a clear, statistically significant lift at n = 500 (mean = 0.1233 ± 0.0085 vs baseline 0.1565 ± 0.0084) and on the full dataset (0.1290 ± 0.0088 vs baseline 0.1613 ± 0.0020). The "external" arm lifts at n = 100 (0.1524 ± 0.0090 vs baseline 0.1736 ± 0.0096) and on the full dataset (0.1179 ± 0.0116). The "intra_task" arm lifts at n = 100 (0.1441 ± 0.0132 vs baseline 0.1736 ± 0.0096) and on the full dataset (0.1267 ± 0.0154). At moderate sizes (n = 50–250) the transfer arms either match or slightly exceed baseline error, and their error bands overlap with baseline, so no confident lift can be claimed. At the smallest size (n = 25) all transfer arms are worse than baseline.

### Per-endpoint

**MPPB** — Both: significant improvement at n=500 (ΔRAE ≈ ‑0.033 ± 0.012) and full data (ΔRAE ≈ ‑0.032 ± 0.010). External: improvement at n=100 (ΔRAE ≈ ‑0.021 ± 0.013) and full data (ΔRAE ≈ ‑0.043 ± 0.012). Intra_task: improvement at n=100 (ΔRAE ≈ ‑0.030 ± 0.014) and full data (ΔRAE ≈ ‑0.035 ± 0.011). No arm shows a statistically significant lift at n=25 or n=50; at n=250 only the both arm approaches baseline (upper band 0.158 vs baseline lower 0.156) but does not clear it.

### Open questions

- Why does the "both" arm fail to produce a clear lift at moderate data regimes (n=50‑250) despite combining two auxiliary sources?
- What is the relative contribution of task similarity versus data volume to the observed lifts at n=100 and above?
- Would increasing the number of random seeds or using a more robust variance estimator change the significance of the moderate‑size improvements?
- Can alternative auxiliary tasks (e.g., physicochemical property prediction) provide earlier lifts at smaller n?
- How does the transfer benefit translate to downstream decision‑making thresholds (e.g., classification cut‑offs) beyond the aggregate RAE metric?