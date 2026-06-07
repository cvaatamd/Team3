# External and Both Transfer Arm Summary

This run completed the recommended external-data extension without overwriting previous results. It evaluates only endpoints with a meaningful registered external source: HLM CLint with Biogen/Fang HLM and Caco-2 Papp A>B with TDC Caco2_Wang.

![Full transfer comparison](full_transfer_arm_comparison.png)

| Target | n | Method | Source setting | Valid seeds | RAE mean ± std | Lift vs baseline |
|---|---:|---|---|---:|---:|---:|
| HLM CLint | 50 | Baseline | target only | 3 | 0.1055 ± 0.0050 | +0.0% |
| HLM CLint | 50 | All intra-task | all other ExpansionRx endpoints | 3 | 0.1038 ± 0.0135 | +1.6% |
| HLM CLint | 50 | External only | Biogen HLM auxiliary | 3 | 0.1542 ± 0.0695 | -46.2% |
| HLM CLint | 50 | Both internal+external | internal auxiliaries + Biogen HLM | 3 | 0.0992 ± 0.0043 | +5.9% |
| HLM CLint | 50 | Domain-selected source | MLM CLint | 3 | 0.1009 ± 0.0036 | +4.4% |
| HLM CLint | full | Baseline | target only | 3 | 0.0850 ± 0.0021 | +0.0% |
| HLM CLint | full | All intra-task | all other ExpansionRx endpoints | 3 | 0.0958 ± 0.0015 | -12.7% |
| HLM CLint | full | External only | Biogen HLM auxiliary | 3 | 0.0910 ± 0.0055 | -7.1% |
| HLM CLint | full | Both internal+external | internal auxiliaries + Biogen HLM | 3 | 0.0899 ± 0.0079 | -5.8% |
| Caco-2 Permeability Papp A>B | 50 | Baseline | target only | 3 | 0.1610 ± 0.0260 | +0.0% |
| Caco-2 Permeability Papp A>B | 50 | All intra-task | all other ExpansionRx endpoints | 3 | 0.1393 ± 0.0145 | +13.5% |
| Caco-2 Permeability Papp A>B | 50 | External only | TDC Caco2_Wang auxiliary | 2 | 0.1962 ± 0.0359 | -21.8% |
| Caco-2 Permeability Papp A>B | 50 | Both internal+external | internal auxiliaries + TDC Caco2_Wang | 3 | 0.1397 ± 0.0054 | +13.3% |
| Caco-2 Permeability Papp A>B | 50 | Domain-selected source | Caco-2 Permeability Efflux | 3 | 0.1411 ± 0.0007 | +12.4% |
| Caco-2 Permeability Papp A>B | full | Baseline | target only | 3 | 0.1103 ± 0.0012 | +0.0% |
| Caco-2 Permeability Papp A>B | full | All intra-task | all other ExpansionRx endpoints | 3 | 0.1488 ± 0.0155 | -35.0% |
| Caco-2 Permeability Papp A>B | full | External only | TDC Caco2_Wang auxiliary | 3 | 0.1422 ± 0.0299 | -28.9% |
| Caco-2 Permeability Papp A>B | full | Both internal+external | internal auxiliaries + TDC Caco2_Wang | 3 | 0.1396 ± 0.0045 | -26.6% |

## Key Interpretation

- **HLM CLint**: external-only is unstable in n=50 and does not improve full-data. The `both` arm is the best HLM n=50 setting in this comparison, but full-data baseline remains strongest or near-strongest.
- **Caco-2 Papp A>B**: TDC external-only is weak and one n=50 seed produced invalid metrics. The `both` arm is stable and useful at n=50, but transfer hurts in full-data.
- **External data quality matters**: both external sources had zero molecule overlap with ExpansionRx, so the code used them as auxiliary heads rather than pooled target labels. This is stricter and more honest, but it also means external data cannot simply increase target-label count.
- **Best story**: transfer is most useful in low-label settings, and `both` is safer than external-only because internal auxiliary tasks stabilize the weaker external signal.

## Invalid Runs

The following run completed but produced NaN metrics and was excluded from `results.parquet` aggregation:

- `Caco-2_Permeability_Papp_AtoB__external__50__s1.json`
