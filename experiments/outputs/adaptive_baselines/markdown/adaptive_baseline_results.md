# Adaptive baseline evaluation

The learned policy and comparison policies are evaluated on the same simulation seeds. The main comparison is learned Q versus uniform random feasible institutional choice; frequency-informed random feasible and shuffled learned sequence are ex-post sequencing diagnostics.

Model: `experiments/outputs/adaptive_planner/model/online_q_table.json`.
Evaluation seeds per scenario: `100`.

## Mean final planner welfare

- Frequency-informed random feasible (`frequency_informed_random`): 20.314
- Learned Q policy (`learned_q`): 20.182
- Uniform random feasible (`random_feasible`): 20.045
- Shuffled learned sequence (`shuffled_learned_sequence`): 20.044
- Permanent bilateral 3-pass (`permanent_bilateral_3pass`): 19.417

## Learned Q minus comparison policies

- vs `frequency_informed_random`: mean difference -0.133, seed-clustered 95% CI [-0.597, 0.324]
- vs `permanent_bilateral_3pass`: mean difference 0.765, seed-clustered 95% CI [0.303, 1.218]
- vs `random_feasible`: mean difference 0.136, seed-clustered 95% CI [-0.336, 0.608]
- vs `shuffled_learned_sequence`: mean difference 0.138, seed-clustered 95% CI [-0.295, 0.555]

## State coverage

- Trained Q states: 3216.
- Unique evaluation states: 1352.
- Unseen evaluation round share: 0.0018.
- Median best-versus-second feasible Q margin: 2.8668.

The pooled comparison does not establish a welfare advantage from the learned state-to-action mapping over simple feasible multi-institution choice.