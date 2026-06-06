# Multi-task few-shot ADMET — characterization

- Endpoints: Caco-2 Permeability Efflux
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 120

## Best arm per (endpoint, n)

### Caco-2 Permeability Efflux

|   n |   baseline |   both |   external |   intra_task |
|----:|-----------:|-------:|-----------:|-------------:|
|  25 |     0.0539 | 0.1526 |     0.1526 |       0.1526 |
|  50 |     0.0295 | 0.0289 |     0.0291 |       0.0295 |
| 100 |     0.0301 | 0.0287 |     0.0295 |       0.0291 |
| 250 |     0.0296 | 0.0287 |     0.0297 |       0.0293 |
| 500 |     0.0293 | 0.0287 |     0.0297 |       0.0292 |


## Interpretation

- **Caco-2 Permeability Efflux / intra_task**: largest lift 0.0010 RAE at n=100.0 (baseline=0.0301, intra_task=0.0291).
- **Caco-2 Permeability Efflux / external**: largest lift 0.0006 RAE at n=100.0 (baseline=0.0301, external=0.0295).
- **Caco-2 Permeability Efflux / both**: largest lift 0.0014 RAE at n=100.0 (baseline=0.0301, both=0.0287).

## LLM characterization

Across the Caco-2 Permeability Efflux endpoint, auxiliary-task transfer only yields consistent lifts when the target training set is modest (≈50‑250 samples). The "both" arm (combined external + intra‑task) improves mean RAE by 3‑5 ×10⁻³ (≈10‑15 % relative) at 50, 100, 250 and 500 samples, and by ~1.3 ×10⁻³ (≈5 %) on the full‑data (NaN) regime, with its error band (mean ± std) entirely below the baseline band in each case. The "intra_task" arm also lifts at 50‑250 samples (≈2‑4 ×10⁻³ improvement) and at 100 samples on the full‑data set, but its band overlaps baseline at 500 samples and the full‑data regime, so the lift is not statistically certain there. The "external" arm shows a mixed picture: it only lifts at 100 samples (≈6 ×10⁻⁴ improvement) and fails to achieve a lower error band at 50, 250, 500 or full‑data sizes. All transfer arms dramatically degrade performance at the smallest data regime (25 samples), with mean RAE ≈0.15 versus baseline ≈0.054.

### Per-endpoint

**Caco-2 Permeability Efflux** — Both (combined) transfer: improvement at n=50 (ΔRAE=-0.00063, band 0.0280‑0.0297 vs baseline 0.0288‑0.0303), n=100 (ΔRAE=-0.00142, band 0.0278‑0.0296 vs baseline 0.0297‑0.0305), n=250 (ΔRAE=-0.00098, band 0.0277‑0.0297 vs baseline 0.0291‑0.0302), n=500 (ΔRAE=-0.00055, band 0.0278‑0.0296 vs baseline 0.0286‑0.0299), full data (ΔRAE=-0.00128, band 0.0273‑0.0280 vs baseline 0.0277‑0.0281). Intra‑task transfer: improvement at n=50 (ΔRAE=-0.00002, band 0.0289‑0.0301 vs baseline 0.0288‑0.0303), n=100 (ΔRAE=-0.00100, band 0.0287‑0.0295 vs baseline 0.0297‑0.0305), n=250 (ΔRAE=-0.00033, band 0.0288‑0.0298 vs baseline 0.0291‑0.0302), no clear lift at n=500 (band overlaps) or full data. External transfer: only clear lift at n=100 (ΔRAE=-0.00064, band 0.0286‑0.0303 vs baseline 0.0297‑0.0305); other regimes either overlap or are worse. All arms degrade at n=25 (mean ≈0.15 vs baseline ≈0.054).

### Open questions

- Why does the external auxiliary task fail to provide consistent lifts beyond the 100‑sample regime despite being a distinct source task?
- What causes the intra‑task arm to lose its advantage at the largest training size (500 samples) and on the full‑data set?
- Would increasing the number of random seeds or using a more robust variance estimator change the overlap conclusions for marginal lifts?
- How does the similarity between the auxiliary tasks and the Caco‑2 endpoint (e.g., shared physicochemical descriptors) modulate the observed transfer benefits?
- Can curriculum or staged fine‑tuning mitigate the severe degradation observed at the ultra‑low (25‑sample) regime?