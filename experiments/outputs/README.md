# Output folders

- `fixed_institution/`: fixed-institution benchmark seed-level data and derived
  tables.
- `composition_sweep/`: behavioral-composition sweep data and tables.
- `adaptive_planner/`: trained Q table and its evaluation output.
- `adaptive_baselines/`: paired evaluation of the Q policy and comparison
  policies.
- `branch_diagnostic/`: counterfactual rollout diagnostic.
- `paper_results/`: compact statistical tables used by the paper.
- `report_figures/`: four main figures generated from the stored results.
- `robustness/`: concise summary of additional learner variants reported in the
  supplement.

Large raw histories are regenerable and are not required for every experiment.
The paper-level analysis uses the compact seed-level outputs wherever possible.
