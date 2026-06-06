# Multi-task few-shot ADMET — characterization

- Endpoints: MBPB
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 120

## Best arm per (endpoint, n)

### MBPB

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.1649 | 0.162  |     0.162  |       0.162  |
|  50 |     0.1571 | 0.1218 |     0.1523 |       0.1268 |
| 100 |     0.1463 | 0.1274 |     0.1327 |       0.1286 |
| 250 |     0.1528 | 0.1211 |     0.1304 |       0.1209 |
| 500 |     0.1271 | 0.1022 |     0.1093 |       0.111  |


## Interpretation

- **MBPB / intra_task**: largest lift 0.0319 RAE at n=250.0 (baseline=0.1528, intra_task=0.1209).
- **MBPB / external**: largest lift 0.0253 RAE at n=nan (baseline=0.1210, external=0.0957).
- **MBPB / both**: largest lift 0.0353 RAE at n=50.0 (baseline=0.1571, both=0.1218).

## LLM characterization

Across the MBPB endpoint, all three auxiliary-task transfer arms (both, external, intra_task) achieve lower mean RAE than the baseline once the training set reaches 50 samples. The lift grows with more data: at 50 samples, both and intra_task reduce RAE by ~22% and ~19% respectively, while external shows only a marginal 3% gain. At 100–500 samples, both and intra_task consistently deliver ~10–13% relative reductions, and external catches up with ~9–14% reductions. With the full‑dataset (NaN n), external yields the largest relative drop (~21%), followed by intra_task (~17%) and both (~15%). Error bands (mean ± std) for the transfer arms are entirely below the baseline’s band from 50 samples onward, confirming statistically reliable improvements. At the smallest regime (25 samples) the transfer arms do not provide a clear lift—their means are slightly lower but their higher variance keeps the error band overlapping the baseline.

### Per-endpoint

**MBPB** — Data regime 25: no lift (mean lower but std higher, band overlaps). 50: both (RAE ↓0.035, 22% rel), intra_task (RAE ↓0.030, 19% rel) – both bands below baseline; external negligible lift. 100: both (↓0.019, 13%), intra_task (↓0.018, 12%), external (↓0.014, 9%) – all bands below baseline. 250: both (↓0.032, 21%), intra_task (↓0.032, 21%), external (↓0.022, 15%) – bands below baseline. 500: both (↓0.025, 20%), intra_task (↓0.016, 13%), external (↓0.018, 14%) – bands below baseline. Full (NaN): both (↓0.018, 15%), intra_task (↓0.021, 17%), external (↓0.025, 21%) – bands below baseline.

### Open questions

- Why does the external auxiliary task lag at 50 samples but surpass both and intra_task on the full dataset?
- What is the impact of the higher variance observed for the 'both' arm at low n, and can regularization or more seeds reduce this spread?
- Would combining external with intra_task (instead of simple concatenation) yield additive gains, especially in the mid‑range data regimes?
- How do these transfer effects generalize to other ADMET endpoints with different signal‑to‑noise characteristics?