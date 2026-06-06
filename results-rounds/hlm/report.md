# Multi-task few-shot ADMET — characterization

- Endpoints: HLM CLint
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 80

## Best arm per (endpoint, n)

### HLM CLint

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.105  | 0.1866 |     0.1866 |       0.1866 |
|  50 |     0.1037 | 0.1162 |     0.1294 |       0.1396 |
| 100 |     0.1063 | 0.1192 |     0.1053 |       0.1305 |
| 250 |     0.1002 | 0.1105 |     0.1055 |       0.1266 |


## Interpretation

- **HLM CLint / intra_task**: largest lift -0.0242 RAE at n=100 (baseline=0.1063, intra_task=0.1305).
- **HLM CLint / external**: largest lift 0.0010 RAE at n=100 (baseline=0.1063, external=0.1053).
- **HLM CLint / both**: largest lift -0.0103 RAE at n=250 (baseline=0.1002, both=0.1105).

## LLM characterization

Across the four training data regimes (n=25, 50, 100, 250) for the HLM CLint endpoint, none of the transfer arms (both, external, intra_task) achieved a statistically reliable reduction in RAE relative to the baseline. The only clear deviation is a degradation for the intra_task arm at n=250, where its entire error band lies above the baseline band. Marginal mean improvements (e.g., external at n=100) are within overlapping error bands and therefore cannot be claimed as lifts.

### Per-endpoint

**HLM CLint** — Baseline RAE bands: 25 → [0.101, 0.109]; 50 → [0.099, 0.108]; 100 → [0.101, 0.112]; 250 → [0.095, 0.105]. Transfer arms: • both – means 0.187, 0.116, 0.119, 0.110 with bands overlapping baseline at all n (no lift). • external – means 0.187, 0.129, 0.105, 0.106; only at n=100 the mean is slightly lower than baseline (0.105 vs 0.106) but its band [0.097, 0.114] overlaps baseline, so no lift; at n=250 the band [0.100, 0.111] also overlaps. • intra_task – means 0.187, 0.140, 0.131, 0.127; bands overlap baseline at n≤100 (no lift) and at n=250 the band [0.107, 0.146] sits entirely above baseline, indicating a degradation.

### Open questions

- Would increasing the number of repeats per arm reduce the high variance observed at low data regimes (n=25, 50)?
- Are there alternative auxiliary tasks (e.g., physicochemical property prediction) that might yield non‑overlapping error bands with baseline?
- How does the choice of multi‑task weighting affect transfer efficacy, especially for the 'both' arm?
- Would larger pre‑training datasets for the external task improve its signal‑to‑noise ratio and produce a clear lift?
- Can per‑compound error analysis reveal systematic biases that are masked by aggregate RAE statistics?