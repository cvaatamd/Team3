# Multi-task few-shot ADMET — characterization

- Endpoints: MBPB
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 80

## Best arm per (endpoint, n)

### MBPB

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.1649 | 0.162  |     0.162  |       0.162  |
|  50 |     0.1571 | 0.12   |     0.1462 |       0.1274 |
| 100 |     0.1463 | 0.1287 |     0.1222 |       0.1173 |
| 250 |     0.1528 | 0.1203 |     0.1139 |       0.1142 |


## Interpretation

- **MBPB / intra_task**: largest lift 0.0386 RAE at n=250 (baseline=0.1528, intra_task=0.1142).
- **MBPB / external**: largest lift 0.0390 RAE at n=250 (baseline=0.1528, external=0.1139).
- **MBPB / both**: largest lift 0.0370 RAE at n=50 (baseline=0.1571, both=0.1200).

## LLM characterization

Across the MBPB endpoint, auxiliary-task transfer only yields statistically clear improvements (mean RAE lower than baseline and the entire arm error band below the baseline band) when the target training set contains at least 50 samples. At n=25, none of the transfer arms (both, external, intra_task) achieve a clear lift. At n=50, the "both" arm improves mean RAE from 0.157 to 0.120 (‑23.6%) and its 95% band (0.103–0.138) sits entirely below the baseline band (0.143–0.171). At n=100, the external and intra_task arms each lift: external drops mean RAE to 0.122 (‑16.5%) with band 0.114–0.130, intra_task to 0.117 (‑19.8%) with band 0.102–0.132, both fully under the baseline band (0.133–0.160). At the largest regime (n=250), all three transfer arms lift: both (0.120, ‑21.2%), external (0.114, ‑25.5%), intra_task (0.114, ‑25.3%) with bands well beneath the baseline band (0.147–0.158). Thus, multi‑task (both) transfer is most effective in the low‑mid regime (50–250), while single‑source external or intra‑task transfer become competitive as more target data are available.

### Per-endpoint

**MBPB** — n=25: No lift (all arms mean ≈0.162‑0.165, bands overlap baseline). n=50: both arm lifts (mean 0.120 vs 0.157, band 0.103‑0.138 < baseline 0.143‑0.171). n=100: external (0.122, band 0.114‑0.130) and intra_task (0.117, band 0.102‑0.132) lift; both arm does not (band overlaps). n=250: both (0.120, band 0.104‑0.137), external (0.114, band 0.102‑0.126), intra_task (0.114, band 0.101‑0.127) all lift.

### Open questions

- Why does the both‑task arm fail to produce a clear lift at n=100 despite lower mean RAE?
- What drives the superior performance of external and intra_task arms at larger data regimes compared to the both arm?
- Would incorporating task similarity metrics or weighting schemes improve transfer effectiveness at the smallest regime (n=25)?
- How robust are these lifts across different random seeds or splits – should we expand the count beyond 5 repeats?
- Can a hybrid schedule (e.g., start with both‑task pretraining then fine‑tune with external data) yield consistent gains across all regimes?