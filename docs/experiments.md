# Experiments and outputs

This file maps the paper's analysis to the scripts and output folders in the
repository.

## Fixed-institution benchmark

**Runner**

```text
experiments/main/run_institution_behavior_analysis.py
```

**Output**

```text
experiments/outputs/fixed_institution/
```

The runner crosses two build modes, seven behavioral populations, eleven
institutions, and 200 matched seeds. The main seed-level output is
`csv/summary_by_seed.csv`. `experiments/analysis/analyze_utility_results.py`
validates the factorial structure and produces the compact tables used for the
fixed-institution results.

The analysis can be rerun without the large per-round history file.

## Behavioral-composition sweeps

**Runner**

```text
experiments/main/run_behavioral_composition_sweep.py
```

**Output**

```text
experiments/outputs/composition_sweep/
```

The experiment varies the number of endpoint-policy agents from zero to five in
the cooperative-to-hoarding and need-based-to-competitive sweeps. The main paper
analysis uses `csv/summary_by_seed.csv` and paired endpoint differences computed
by `experiments/analysis/analyze_paper_results.py`.

## Adaptive Q-learning planner

**Runner**

```text
experiments/adaptive/run_online_q_planner.py
```

**Output**

```text
experiments/outputs/adaptive_planner/
```

The default runner parameters match the specification reported in the paper. It
trains the Q table, evaluates the frozen policy on 100 seeds in each of six
behavioral scenarios, and writes the model plus seed-level and round-level
evaluation outputs.

`experiments/analysis/analyze_online_q_planner.py` produces descriptive tables
for the learned policy, including action allocation and capacity use.

## Adaptive baseline evaluation

**Runner**

```text
experiments/adaptive/evaluate_online_planner_baselines.py
```

**Output**

```text
experiments/outputs/adaptive_baselines/
```

The learned policy is compared with:

- permanent bilateral 3-pass,
- uniform random feasible institutional choice,
- frequency-informed random feasible choice,
- a shuffled learned action sequence.

All policies use matched seed-scenario games. The pooled final-welfare intervals
reported in the paper use 10,000 seed-cluster bootstrap draws; scenario-specific
paired intervals use 2,000 draws.

## Counterfactual branch diagnostic

**Runner**

```text
experiments/adaptive/run_branch_diagnostic.py
```

**Output**

```text
experiments/outputs/branch_diagnostic/
```

The runner samples 54 checkpoints from 18 base games. Every feasible first action
is evaluated with 40 continuations. The first 20 continuations select the
provisional best and second-best actions, while the remaining 20 evaluate those
choices independently. Aggregate uncertainty is clustered at the base-game
level.

## Paper-level analysis

**Analysis script**

```text
experiments/analysis/analyze_paper_results.py
```

**Output**

```text
experiments/outputs/paper_results/
```

This is the final statistical assembly step. It reads the experiment outputs and
produces the compact tables for:

- pooled adaptive-policy comparisons,
- scenario-specific Q-versus-random comparisons,
- action shares,
- behavioral-composition endpoint sensitivity,
- central-versus-equity comparisons,
- the pooled build-mode comparison,
- evaluation-state coverage.

## Figures

**Script**

```text
experiments/visualization/make_report_figures.py
```

**Output**

```text
experiments/outputs/report_figures/
```

The script creates four figures corresponding to the main paper structure:

1. fixed-institution benchmark heatmap,
2. behavioral-composition sweeps and central-versus-equity comparison,
3. learned Q-policy action allocation,
4. paired adaptive-policy welfare comparisons.

## Additional learner variants

The supplement reports three additional learner checks: a modified State-v2
representation, longer-return updates, and a larger State-v3 representation.
Their concise outcomes are stored in
`experiments/outputs/robustness/learner_variant_summary.csv`. They are not part of
the main reproduction sequence because none replaces the reported learner or
changes the paper's main inference.
