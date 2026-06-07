# Selected Source Validation Summary

This report uses the pairwise transfer matrix to make source-selection decisions, then adds one new selected-subset run for MBPB full data (`MPPB+MGMB`).

## Decision Table

| target          | regime   | recommended                | avoid              | decision                                        |
|:----------------|:---------|:---------------------------|:-------------------|:------------------------------------------------|
| HLM CLint       | n=50     | MLM CLint                  | MBPB               | Use metabolism-related source only in few-shot. |
| HLM CLint       | full     | baseline / no transfer     | MLM, MBPB          | Baseline already strong; transfer adds noise.   |
| MBPB            | n=50     | MPPB                       | HLM CLint          | Plasma/brain binding relation is useful.        |
| MBPB            | full     | MPPB; MPPB+MGMB comparable | none clear         | Binding/tissue sources consistently help.       |
| Caco-2 Papp A>B | n=50     | Caco-2 Efflux              | MBPB               | Permeability-related source helps in few-shot.  |
| Caco-2 Papp A>B | full     | baseline / no transfer     | Efflux, LogD, MBPB | Full target data beats tested transfer sources. |

## Validation Metrics

| target_endpoint              | n    | role                       | source                     |   valid_seeds |   rae_mean |   rae_std |   baseline_rae |   lift_vs_baseline |   spearman_mean |
|:-----------------------------|:-----|:---------------------------|:---------------------------|--------------:|-----------:|----------:|---------------:|-------------------:|----------------:|
| HLM CLint                    | 50   | baseline                   | baseline                   |             3 |     0.1055 |    0.005  |         0.1055 |             0      |          0.3339 |
| HLM CLint                    | 50   | selected                   | MLM CLint                  |             3 |     0.1009 |    0.0036 |         0.1055 |             0.0438 |          0.3982 |
| HLM CLint                    | 50   | negative_control           | MBPB                       |             3 |     0.1856 |    0.1226 |         0.1055 |            -0.7593 |          0.2596 |
| HLM CLint                    | full | baseline                   | baseline                   |             3 |     0.085  |    0.0021 |         0.085  |             0      |          0.5657 |
| HLM CLint                    | full | avoid_transfer_high_source | MLM CLint                  |             3 |     0.0961 |    0.0067 |         0.085  |            -0.1304 |          0.4925 |
| MBPB                         | 50   | baseline                   | baseline                   |             3 |     0.1472 |    0.0007 |         0.1472 |             0      |          0.5603 |
| MBPB                         | 50   | selected                   | MPPB                       |             3 |     0.1295 |    0.0089 |         0.1472 |             0.12   |          0.7004 |
| MBPB                         | 50   | negative_control           | HLM CLint                  |             3 |     0.1516 |    0.0103 |         0.1472 |            -0.0301 |          0.4778 |
| MBPB                         | full | baseline                   | baseline                   |             3 |     0.1206 |    0.0127 |         0.1206 |             0      |          0.7541 |
| MBPB                         | full | selected_single            | MPPB                       |             3 |     0.0948 |    0.005  |         0.1206 |             0.2139 |          0.8183 |
| MBPB                         | full | selected_subset            | MPPB+MGMB                  |             3 |     0.0959 |    0.0034 |         0.1206 |             0.2051 |          0.8099 |
| MBPB                         | full | related_single             | MGMB                       |             3 |     0.097  |    0.0044 |         0.1206 |             0.196  |          0.8016 |
| Caco-2 Permeability Papp A>B | 50   | baseline                   | baseline                   |             3 |     0.161  |    0.026  |         0.161  |             0      |          0.2337 |
| Caco-2 Permeability Papp A>B | 50   | selected                   | Caco-2 Permeability Efflux |             3 |     0.1411 |    0.0007 |         0.161  |             0.1237 |          0.2878 |
| Caco-2 Permeability Papp A>B | 50   | negative_control           | MBPB                       |             3 |     0.1661 |    0.0243 |         0.161  |            -0.0312 |          0.0959 |
| Caco-2 Permeability Papp A>B | full | baseline                   | baseline                   |             3 |     0.1103 |    0.0012 |         0.1103 |             0      |          0.5553 |
| Caco-2 Permeability Papp A>B | full | avoid_transfer_high_source | Caco-2 Permeability Efflux |             3 |     0.1407 |    0.0089 |         0.1103 |            -0.276  |          0.4699 |

## Figure

![Selected source validation bars](selected_source_validation_bars.png)
