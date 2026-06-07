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
|  25 |     0.4136 | 0.7927 |     0.5786 |
|  50 |     0.424  | 1.389  |     1.538  |
| 100 |     0.4272 | 9.9585 |     2.4048 |

### MBPB

|   n |   baseline |   both |   external |
|----:|-----------:|-------:|-----------:|
|  25 |     0.1649 | 0.1502 |     0.1891 |
|  50 |     0.1468 | 0.1624 |     0.2006 |
| 100 |     0.1463 | 0.1513 |     0.1686 |

### MPPB

|   n |   baseline |   both |   external |
|----:|-----------:|-------:|-----------:|
|  25 |     0.181  | 0.1837 |     0.2057 |
|  50 |     0.1703 | 0.1954 |     0.2205 |
| 100 |     0.1736 | 0.1668 |     0.1718 |


## Interpretation

- **KSOL / external**: largest lift -0.1650 RAE at n=25 (baseline=0.4136, external=0.5786).
- **KSOL / both**: largest lift -0.3791 RAE at n=25 (baseline=0.4136, both=0.7927).
- **MBPB / external**: largest lift -0.0223 RAE at n=100 (baseline=0.1463, external=0.1686).
- **MBPB / both**: largest lift 0.0147 RAE at n=25 (baseline=0.1649, both=0.1502).
- **MPPB / external**: largest lift 0.0018 RAE at n=100 (baseline=0.1736, external=0.1718).
- **MPPB / both**: largest lift 0.0068 RAE at n=100 (baseline=0.1736, both=0.1668).