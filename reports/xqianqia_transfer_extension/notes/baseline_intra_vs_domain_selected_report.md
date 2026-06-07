# Baseline / Intra-task vs Domain-selected Source Comparison

This run adds matched **baseline** and **intra_task** arms for three target endpoints using 20 Chemprop epochs for transfer arms and 3 seeds. It is stored separately from the earlier pairwise and selected-source experiments.

![Comparison](arm_vs_domain_selected_comparison.png)

| Target | n | Method | Source setting | RAE mean ± std | Lift vs baseline |
|---|---:|---|---|---:|---:|
| HLM CLint | 50 | All intra-task | all other ExpansionRx endpoints | 0.1038 ± 0.0135 | +1.6% |
| HLM CLint | 50 | Baseline | target only | 0.1055 ± 0.0050 | +0.0% |
| HLM CLint | 50 | Domain-selected source | MLM CLint | 0.1009 ± 0.0036 | +4.4% |
| HLM CLint | full | All intra-task | all other ExpansionRx endpoints | 0.0958 ± 0.0015 | -12.7% |
| HLM CLint | full | Baseline | target only | 0.0850 ± 0.0021 | +0.0% |
| MBPB | 50 | All intra-task | all other ExpansionRx endpoints | 0.1305 ± 0.0165 | +11.4% |
| MBPB | 50 | Baseline | target only | 0.1472 ± 0.0007 | +0.0% |
| MBPB | 50 | Domain-selected source | MPPB | 0.1295 ± 0.0089 | +12.0% |
| MBPB | full | All intra-task | all other ExpansionRx endpoints | 0.0975 ± 0.0148 | +19.1% |
| MBPB | full | Baseline | target only | 0.1206 ± 0.0127 | +0.0% |
| MBPB | full | Domain-selected source | MPPB | 0.0948 ± 0.0050 | +21.4% |
| Caco-2 Permeability Papp A>B | 50 | All intra-task | all other ExpansionRx endpoints | 0.1393 ± 0.0145 | +13.5% |
| Caco-2 Permeability Papp A>B | 50 | Baseline | target only | 0.1610 ± 0.0260 | +0.0% |
| Caco-2 Permeability Papp A>B | 50 | Domain-selected source | Caco-2 Permeability Efflux | 0.1411 ± 0.0007 | +12.4% |
| Caco-2 Permeability Papp A>B | full | All intra-task | all other ExpansionRx endpoints | 0.1488 ± 0.0155 | -35.0% |
| Caco-2 Permeability Papp A>B | full | Baseline | target only | 0.1103 ± 0.0012 | +0.0% |

## Take-home interpretation

- **HLM CLint**: all-intra-task gives only a tiny few-shot gain and hurts in full-data. The domain-selected single source (MLM) was slightly better at n=50 but should be avoided in full-data.
- **MBPB**: transfer is consistently useful. All-intra-task improves over baseline, while the domain-selected source MPPB is at least as strong and more interpretable; adding MGMB did not beat MPPB alone.
- **Caco-2 Papp A>B**: all-intra-task and Efflux both help in n=50, but transfer hurts in full-data. This supports using transfer mainly in low-label settings for this endpoint.

Overall, the arm-level experiment answers **whether transfer helps**, while the source-level/domain experiment explains **which source caused the gain or negative transfer**.
