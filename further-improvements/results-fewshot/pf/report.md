# Multi-task few-shot ADMET — characterization

- Endpoints: MBPB, MPPB, KSOL
- Arms: baseline, external, both
- n grid: [25, 50, 100]
- seeds: [0, 1, 2, 3, 4]
- jobs run: 135
- metric: RAE (lower is better)

## Best arm per (endpoint, n)

### KSOL

|   n |   baseline |   both |   external |
|----:|-----------:|-------:|-----------:|
|  25 |     0.4136 | 0.3613 |     0.3725 |
|  50 |     0.424  | 0.3795 |     0.3859 |
| 100 |     0.4272 | 0.4094 |     0.3807 |

### MBPB

|   n |   baseline |   both |   external |
|----:|-----------:|-------:|-----------:|
|  25 |     0.1649 | 0.1159 |     0.1394 |
|  50 |     0.1468 | 0.1216 |     0.1422 |
| 100 |     0.1463 | 0.1278 |     0.1235 |

### MPPB

|   n |   baseline |   both |   external |
|----:|-----------:|-------:|-----------:|
|  25 |     0.181  | 0.1441 |     0.1701 |
|  50 |     0.1703 | 0.1343 |     0.1784 |
| 100 |     0.1736 | 0.1307 |     0.158  |


## Interpretation

- **KSOL / external**: largest lift 0.0465 RAE at n=100 (baseline=0.4272, external=0.3807).
- **KSOL / both**: largest lift 0.0523 RAE at n=25 (baseline=0.4136, both=0.3613).
- **MBPB / external**: largest lift 0.0255 RAE at n=25 (baseline=0.1649, external=0.1394).
- **MBPB / both**: largest lift 0.0490 RAE at n=25 (baseline=0.1649, both=0.1159).
- **MPPB / external**: largest lift 0.0156 RAE at n=100 (baseline=0.1736, external=0.1580).
- **MPPB / both**: largest lift 0.0429 RAE at n=100 (baseline=0.1736, both=0.1307).