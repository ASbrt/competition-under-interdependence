# Institutional Coordination under Interdependence

This repository contains the simulation code and analysis for my master's course
paper on exchange institutions, behavioral heterogeneity and scarce
coordination capacity.

The project studies a small resource-interdependent economy. Five agents have
asymmetric access to five resources, while development requires complementary
resource bundles. This makes exchange useful, but the outcome of exchange also
depends on how agents behave and on the rules through which resources are
matched or transferred.

The analysis has two parts: The first compares fixed exchange institutions under
different behavioral populations. The second introduces a limited coordination
budget and asks whether a tabular Q-learning planner can use state information
to decide when stronger institutions should be deployed.

The model is deliberately stylized. It is a computational testbed and pilot, not an
estimate of a particular type of economy or institution.

## Research questions

1. When resource-interdependent agents follow heterogeneous decision policies,
   how do alternative exchange protocols affect development?
2. Can a simple tabular baseline planner exploit state information when stronger
   coordination is made scarce?

## Repository layout

```text
engine/                     simulation mechanics, policies, and institutions
experiments/
  main/                     fixed-institution and composition experiments
  adaptive/                 scarce-capacity environment and Q-planner evaluation
  analysis/                 post-processing and paper-level statistical analysis
  visualization/            the four main paper figures
  outputs/                  stored outputs used by the paper
docs/
  model.md                  model and planner definitions
  experiments.md            what each experiment does and produces
  reproducibility.md        commands for reproduction
submission/                 the final submitted paper as well as my presentation slides from an earlier presentation on the project
```

The main output folders follow the same sequence as the analysis:

```text
experiments/outputs/fixed_institution/
experiments/outputs/composition_sweep/
experiments/outputs/adaptive_planner/
experiments/outputs/adaptive_baselines/
experiments/outputs/branch_diagnostic/
experiments/outputs/paper_results/
experiments/outputs/report_figures/
```

Each folder has one role. Each experiment has one output location, so the analysis scripts and stored results use the same directory structure.

## Experimental sequence

### 1. Fixed-institution benchmark

`experiments/main/run_institution_behavior_analysis.py` crosses:

- 2 build modes,
- 7 behavioral populations,
- 11 exchange institutions, and
- 200 matched simulation seeds.

This gives 30,800 games. The main analysis uses the development-oriented slice,
while the crown-aware build mode is reported as a secondary comparison.

### 2. Behavioral-composition sweeps

`experiments/main/run_behavioral_composition_sweep.py` varies the number of
agents using the endpoint policy along two five-agent gradients:

- cooperative → hoarding,
- need-based → competitive.

Each composition-institution cell is evaluated on 100 matched seeds.

### 3. Adaptive institutional planner

`experiments/adaptive/run_online_q_planner.py` trains a tabular Q-learning
planner in an environment where stronger coordination uses a regenerating
capacity budget. The default configuration is the specification reported in the
paper: 12,000 training episodes, 20 rounds, `lambda = 0.25`, maximum
coordination capacity 8 and 100 evaluation seeds per behavioral scenario.

### 4. Paired adaptive-policy evaluation

`experiments/adaptive/evaluate_online_planner_baselines.py` evaluates the learned
policy and comparison policies on the same seed-scenario games. The main
independent comparator is uniform random feasible institutional choice.
Frequency-informed random feasible and the shuffled learned sequence are
sequencing diagnostics that remove the learned state-to-action mapping in
different ways.

For pooled comparisons, the six scenarios belonging to one simulation seed are
kept together in a 10,000-draw seed-cluster bootstrap. Scenario-specific
intervals use 2,000 paired bootstrap draws.

### 5. Counterfactual branch diagnostic

`experiments/adaptive/run_branch_diagnostic.py` deep-copies selected simulated
states, tries each feasible first institution, and evaluates the resulting
continuations. The diagnostic uses 18 base games, checkpoints at rounds 3, 10,
and 16, and separate rollout samples for selecting and evaluating the apparent
best action. It is exploratory and is interpreted as evidence about
continuation-conditioned first-action values.

### 6. Paper tables and figures

`experiments/analysis/analyze_paper_results.py` produces the compact tables used
for the reported comparisons. `experiments/visualization/make_report_figures.py`
then produces the four main figures from those stored tables and experiment
outputs.

## Setup

Python 3.12--3.14 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Reproducing the analysis

The complete reproduction sequence, including regeneration of the analyses from stored outputs and the more expensive full simulation reruns, is documented in [docs/reproducibility.md](docs/reproducibility.md).

If the stored experiment outputs are already present, the paper tables and
figures can be regenerated directly:

```bash
python3 experiments/analysis/analyze_utility_results.py
python3 experiments/analysis/analyze_online_q_planner.py
python3 experiments/analysis/analyze_paper_results.py
python3 experiments/visualization/make_report_figures.py
```

The expensive simulation and training commands are listed separately in the
reproducibility guide.

## Main interpretation

The fixed experiments show large differences across complete
institution-behavior packages. Behavioral composition affects decentralized and
voluntary mechanisms particularly strongly in the two composition sweeps, while
central mechanisms respond differently to the two behavioral transitions. The
central-versus-equity comparison also produces an in-model trade-off between
aggregate development and the final score gap.

In the adaptive extension, the learned Q policy does not show a statistically
distinguishable welfare advantage over uniform random feasible institutional
choice at the reported Monte Carlo precision. Frequency-informed and shuffled
sequence diagnostics lead to the same cautious interpretation. The branch
rollouts nevertheless show that alternative feasible first actions can have
different continuation-conditioned values at sampled checkpoints. The result is
therefore not that institutional choice is irrelevant, but that the tested
state representation and Q table do not establish a reliable welfare gain from
the learned state-to-action mapping.

More detail on the model is in [docs/model.md](docs/model.md), and the experiment
files are mapped to their outputs in [docs/experiments.md](docs/experiments.md).
