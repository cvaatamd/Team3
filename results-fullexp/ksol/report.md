# Multi-task few-shot ADMET — characterization

- Endpoints: KSOL
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 120

## Best arm per (endpoint, n)

### KSOL

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.4136 | 0.4286 |     0.4286 |       0.4286 |
|  50 |     0.424  | 0.4021 |     0.4004 |       0.4217 |
| 100 |     0.4272 | 0.3874 |     0.4021 |       0.5433 |
| 250 |     0.3961 | 0.3737 |     0.3308 |       0.3548 |
| 500 |     0.3724 | 0.343  |     0.341  |       0.3447 |


## Interpretation

- **KSOL / intra_task**: largest lift 0.0413 RAE at n=250.0 (baseline=0.3961, intra_task=0.3548).
- **KSOL / external**: largest lift 0.0653 RAE at n=250.0 (baseline=0.3961, external=0.3308).
- **KSOL / both**: largest lift 0.0420 RAE at n=nan (baseline=0.3289, both=0.2869).

## LLM characterization

For the KSOL endpoint, only the external auxiliary task consistently yields a statistically reliable lift (lower mean RAE with its entire error band below the baseline) in two regimes: (1) at a moderate training set size of 250 samples, where RAE drops from 0.396 ± 0.025 (baseline) to 0.331 ± 0.020 (external), a 16.5% relative reduction; and (2) on the full‑data regime (n ≈ all), where RAE falls from 0.329 ± 0.008 to 0.270 ± 0.013, an 18.0% reduction. The "both" arm (combining external and intra‑task auxiliaries) only shows a reliable lift on the full‑data regime, decreasing RAE to 0.287 ± 0.009 (≈ 12.8% reduction). The intra‑task auxiliary never achieves a band‑wise improvement and even degrades at n = 100. In smaller data regimes (n ≤ 100) all transfer arms have overlapping error bands with the baseline, indicating no confident benefit.

### Per-endpoint

**KSOL** — - n=25: No arm improves (all bands overlap baseline).\n- n=50: No arm improves (bands overlap).\n- n=100: No arm improves (bands overlap).\n- n=250: External auxiliary improves (0.331 ± 0.020 vs 0.396 ± 0.025, ΔRAE = -0.065, -16.5%). Both and intra‑task do not improve.\n- n=500: No arm achieves a full‑band improvement (all overlap).\n- Full data (NaN): Both (0.287 ± 0.009, ΔRAE = -0.042, -12.8%) and External (0.270 ± 0.013, ΔRAE = -0.059, -18.0%) improve. Intra‑task (0.310 ± 0.014) does not improve.

### Open questions

- Why does the external auxiliary only become effective at n≥250? Investigate the representation overlap between external pre‑training data and KSOL chemistry.
- Can a larger or more diverse external dataset shift the lift to smaller n regimes?
- Would alternative weighting or curriculum strategies for the intra‑task auxiliary reduce its variance and enable a lift?
- Is the observed lift on the full‑data regime driven by regularization effects of multi‑task training? Test with ablations that remove the auxiliary loss after a certain epoch.
- Assess whether combining external data with task‑specific fine‑tuning (e.g., freezing early layers) yields stronger gains at low‑sample sizes.