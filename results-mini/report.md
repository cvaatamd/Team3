# Multi-task few-shot ADMET — characterization

- Endpoints: HLM CLint, MBPB
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 8

## Best arm per (endpoint, n)

### HLM CLint

|   n |   baseline |   intra_task |
|----:|-----------:|-------------:|
|  50 |     0.1036 |       0.1271 |
| 250 |     0.0978 |       0.101  |


## Interpretation

- **HLM CLint / intra_task**: largest lift -0.0032 RAE at n=250 (baseline=0.0978, intra_task=0.1010).

## LLM characterization

Across the two data regimes examined (n=50 and n=250), the intra‑task auxiliary training arm does not produce a statistically reliable lift over the baseline for the HLM CLint endpoint. At n=50 the intra‑task mean (0.127) is higher than baseline (0.104) but its 95% confidence interval (0.071–0.183) fully overlaps the baseline interval (0.095–0.112). At n=250 the intra‑task mean (0.101) is marginally above baseline (0.098) yet the confidence intervals (intra‑task 0.093–0.109, baseline 0.086–0.109) are indistinguishable. Consequently, in the current experimental setup, intra‑task transfer yields no clear benefit in either low‑ or moderate‑size training sets.

### Per-endpoint

**HLM CLint** — Baseline (n=50): 0.104 ± 0.008 (95% CI). Intra‑task (n=50): 0.127 ± 0.056. No significant difference. Baseline (n=250): 0.098 ± 0.012. Intra‑task (n=250): 0.101 ± 0.008. Difference of +0.003 is within the error band.

### Open questions

- Would auxiliary tasks that are more chemically or mechanistically related to HLM clearance (e.g., CYP450 metabolism assays) produce a measurable transfer benefit?
- How does increasing the number of random seeds (beyond the current count=2) tighten the confidence intervals and affect significance testing?
- Does model capacity (e.g., larger graph neural networks) interact with auxiliary task transfer, especially in the low‑data regime?
- Can multi‑task learning with a diverse set of ADMET endpoints improve overall representation quality and indirectly boost HLM CLint performance?
- What is the impact of data augmentation (e.g., conformer ensembles, SMILES enumeration) on the efficacy of auxiliary transfer for this endpoint?