# Multi-task few-shot ADMET — characterization

- Endpoints: HLM CLint, MBPB
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 240

## Best arm per (endpoint, n)

### HLM CLint

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.105  | 0.1866 |     0.1866 |       0.1866 |
|  50 |     0.1037 | 0.1119 |     0.1271 |       0.1488 |
| 100 |     0.1063 | 0.1434 |     0.1071 |       0.1222 |
| 250 |     0.1002 | 0.111  |     0.1028 |       0.1183 |
| 500 |     0.0926 | 0.1103 |     0.0964 |       0.1253 |

### MBPB

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.1649 | 0.162  |     0.162  |       0.162  |
|  50 |     0.1525 | 0.1165 |     0.1432 |       0.1219 |
| 100 |     0.1463 | 0.1194 |     0.1245 |       0.1252 |
| 250 |     0.1528 | 0.112  |     0.1197 |       0.112  |
| 500 |     0.1271 | 0.1013 |     0.1204 |       0.107  |


## Interpretation

- **HLM CLint / intra_task**: largest lift -0.0159 RAE at n=100.0 (baseline=0.1063, intra_task=0.1222).
- **HLM CLint / external**: largest lift -0.0008 RAE at n=100.0 (baseline=0.1063, external=0.1071).
- **HLM CLint / both**: largest lift -0.0081 RAE at n=50.0 (baseline=0.1037, both=0.1119).
- **MBPB / intra_task**: largest lift 0.0408 RAE at n=250.0 (baseline=0.1528, intra_task=0.1120).
- **MBPB / external**: largest lift 0.0331 RAE at n=250.0 (baseline=0.1528, external=0.1197).
- **MBPB / both**: largest lift 0.0408 RAE at n=250.0 (baseline=0.1528, both=0.1120).

## LLM characterization

Across the two ADMET endpoints, only the MBPB assay benefits from auxiliary-task pretraining. For MBPB, the "both" (combined external+intra) and "intra_task" arms consistently achieve lower error than the baseline once the target training set reaches 50 compounds, with mean reductions of 0.03–0.04 units and non‑overlapping mean±std bands (e.g., n=50: baseline 0.1525±0.0070 vs. both 0.1165±0.0176). The "external" arm sometimes improves (n=100, 250, full) but its confidence intervals overlap the baseline at the smallest and largest data points, so its benefit is ambiguous. For the HLM CLint assay, none of the auxiliary tasks improve performance; all transfer arms have higher mean errors than baseline at every data regime, and the differences are well within the combined error bands, indicating no statistically significant lift. In the high‑data regime (n=500) the baseline still outperforms all transfer arms (e.g., baseline 0.0927±0.0046 vs. both 0.1103±0.0113).

### Per-endpoint

**HLM CLint** — Baseline error declines from 0.105±0.004 (n=25) to 0.0848±0.0016 (full). All transfer arms (both, external, intra_task) are higher at every n (e.g., both 0.111±0.015 at n=50) and their 95% error bands overlap the baseline, indicating no useful transfer.

**MBPB** — Baseline error drops from 0.165±0.010 (n=25) to 0.121±0.011 (full). Both and intra_task arms achieve lower errors from n=50 onward (e.g., both 0.1165±0.0176 vs. baseline 0.1525±0.0070 at n=50) with non‑overlapping bands, confirming significant transfer. External shows mixed results: clear improvement at n=100 (0.1245±0.0066) and n=250 (0.1197±0.0183), but overlapping bands at n=25, 50, 500, and full.

### Open questions

- What molecular features drive the strong transfer to MBPB but not to HLM CLint? Investigate task similarity metrics.
- Can alternative auxiliary tasks (e.g., physicochemical property prediction) provide lift for HLM CLint?
- Would larger ensembles or variance reduction techniques sharpen the confidence intervals for the external arm on MBPB?
- Is the observed benefit for MBPB robust across different model architectures (e.g., GNN vs. transformer)?
- How does the choice of pretraining data size affect transfer efficacy in the low‑data regime (n<50)?