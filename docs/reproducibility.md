# Reproducibility

The repository includes the compact outputs used by the paper, so the analysis
and figures can be regenerated without retraining the planner. Full simulation
reruns are also possible but are substantially more expensive.

All commands below assume the repository root as the working directory.

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Supported Python versions are defined in `pyproject.toml`.

## 2. Regenerate tables and figures from stored outputs

This is the fastest way to verify the reported analysis:

```bash
python3 experiments/analysis/analyze_utility_results.py
python3 experiments/analysis/analyze_online_q_planner.py
python3 experiments/analysis/analyze_paper_results.py
python3 experiments/visualization/make_report_figures.py
```

The four main figures are written to
`experiments/outputs/report_figures/`.

## 3. Rerun the fixed-institution benchmark

The defaults already match the paper specification:

```bash
python3 experiments/main/run_institution_behavior_analysis.py
```

Equivalent explicit settings are:

```bash
N_SEEDS=200 ROUNDS=20 MAX_BUILDS_PER_AGENT_PER_ROUND=4 \
python3 experiments/main/run_institution_behavior_analysis.py
```

The runner writes to `experiments/outputs/fixed_institution/` and replaces the
source CSVs in that folder. Run the fixed-institution analysis afterward.

## 4. Rerun the behavioral-composition sweeps

```bash
python3 experiments/main/run_behavioral_composition_sweep.py
```

The default run uses:

- development-oriented building,
- seeds 0--99,
- restrictive counts 0--5,
- both behavioral sweeps,
- all six composition-sweep institutions.

It writes to `experiments/outputs/composition_sweep/`.

## 5. Retrain and evaluate the Q-learning planner

The default configuration in `run_online_q_planner.py` is the paper
specification:

```text
training episodes             12000
training random seed          20260717
training seed offset          100000
evaluation seed offset        500000
evaluation seeds/scenario     100
rounds                        20
maximum coordination capacity 8
capacity recovery             1
lambda                        0.25
alpha start                   0.25
alpha floor                   0.01
gamma                         1.0
epsilon start                 1.0
epsilon end                   0.03
epsilon decay fraction        0.90
```

A full retraining run is:

```bash
python3 experiments/adaptive/run_online_q_planner.py
```

This overwrites `experiments/outputs/adaptive_planner/`. The run is seeded and is
intended to be deterministic for the numerical outputs given the same code and
dependency versions. Plot rendering can differ slightly across systems.

## 6. Rerun the paired adaptive-policy evaluation

After the Q model exists:

```bash
python3 experiments/adaptive/evaluate_online_planner_baselines.py
```

Defaults:

```text
evaluation seeds/scenario       100
scenario bootstrap draws       2000
pooled seed-cluster draws     10000
```

The script writes to `experiments/outputs/adaptive_baselines/`.

## 7. Rerun the branch diagnostic

```bash
python3 experiments/adaptive/run_branch_diagnostic.py
```

This diagnostic is computationally expensive because each sampled checkpoint is
continued many times under alternative first actions and continuation policies.
It writes to `experiments/outputs/branch_diagnostic/`.

## 8. Final assembly

After any full reruns, regenerate the paper tables and figures:

```bash
python3 experiments/analysis/analyze_paper_results.py
python3 experiments/visualization/make_report_figures.py
```

The resulting `paper_results/` tables are the intended source for reported
confidence intervals and the policy-comparison figure.
