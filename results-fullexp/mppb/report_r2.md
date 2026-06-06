# Multi-task few-shot ADMET — characterization

- Endpoints: MPPB
- Arms: baseline, intra_task, external, both
- n grid: [25, 50, 100, 250, 500, None]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 120
- metric: R2 (higher is better)

## Best arm per (endpoint, n)

### MPPB

|   n |   baseline |    both |   external |   intra_task |
|----:|-----------:|--------:|-----------:|-------------:|
|  25 |    -0.3336 | -0.6168 |    -0.6168 |      -0.6168 |
|  50 |    -0.1678 |  0.0209 |    -0.2802 |      -0.0626 |
| 100 |    -0.2541 |  0.0539 |     0.0507 |       0.0771 |
| 250 |    -0.1285 |  0.1654 |     0.1033 |       0.0357 |
| 500 |     0.0025 |  0.2958 |     0.2101 |       0.141  |


## Interpretation

- **MPPB / intra_task**: largest lift 0.3785 R2 at n=nan (baseline=-0.1200, intra_task=0.2585).
- **MPPB / external**: largest lift 0.4978 R2 at n=nan (baseline=-0.1200, external=0.3778).
- **MPPB / both**: largest lift 0.3374 R2 at n=nan (baseline=-0.1200, both=0.2174).