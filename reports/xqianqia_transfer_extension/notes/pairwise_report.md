# Pairwise Source-Target Transfer Summary

This focused experiment tests which individual source endpoints help each target endpoint.

Lift is computed as `(baseline RAE - transfer RAE) / baseline RAE`; positive values mean the source helps.

## Best Pairwise Source

| target_endpoint              | n    | source                     | priority    |   rae_mean |   baseline_rae | lift_vs_baseline   |   valid_seeds |
|:-----------------------------|:-----|:---------------------------|:------------|-----------:|---------------:|:-------------------|--------------:|
| Caco-2 Permeability Papp A>B | 50   | Caco-2 Permeability Efflux | high        |  0.141103  |      0.161023  | +12.4%             |             3 |
| Caco-2 Permeability Papp A>B | full | Caco-2 Permeability Efflux | high        |  0.14071   |      0.110272  | -27.6%             |             3 |
| HLM CLint                    | 50   | MLM CLint                  | high        |  0.100879  |      0.105495  | +4.4%              |             3 |
| HLM CLint                    | full | MBPB                       | low_control |  0.0897267 |      0.0849863 | -5.6%              |             3 |
| MBPB                         | 50   | MPPB                       | high        |  0.129542  |      0.147211  | +12.0%             |             3 |
| MBPB                         | full | MPPB                       | high        |  0.0948087 |      0.120603  | +21.4%             |             3 |

## Output Files

- `pairwise_runs.csv`: per-seed raw results
- `pairwise_summary.csv`: aggregated metrics and lift values
- `best_pairwise_source.csv`: best source per target and n
- `transfer_lift_heatmap_n50.png` and `transfer_lift_heatmap_nfull.png`: heatmaps