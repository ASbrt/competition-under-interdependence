# Counterfactual branch diagnostic

Verdict: **MIXED / INCONCLUSIVE**.

## Design

- Base games: 18 using seeds 700000, 700001, 700002, 700100, 700101, 700102, 700200, 700201, 700202, 700300, 700301, 700302, 700400, 700401, 700402, 700500, 700501, 700502.
- Checkpoints: 54 at rounds (3, 10, 16).
- Continuations per feasible action: 40.
- Selection/evaluation split: 20 / 20.
- Candidate branches are deep-copied from the same checkpoint. For a checkpoint, continuation policy, and replication index, every feasible first action uses the same production RNG state at the branch point and the same policy RNG seed pattern.
- Branches are not forced to stay identical after actions diverge; switching costs, workload, future capacity, and later feasibility are recalculated normally.

## Aggregate gaps

| continuation_policy       | comparison                          |   mean_difference |   base_game_clustered_ci95_low |   base_game_clustered_ci95_high |   resolved_share |   resolved_count |   checkpoints |   base_games |   q_equals_selection_best_share |
|:--------------------------|:------------------------------------|------------------:|-------------------------------:|--------------------------------:|-----------------:|-----------------:|--------------:|-------------:|--------------------------------:|
| learned_q                 | q_chosen_minus_bilateral            |             0.116 |                         -0.515 |                           0.698 |            0.315 |               17 |            54 |           18 |                           0.574 |
| learned_q                 | selected_best_minus_bilateral       |             1.329 |                          0.811 |                           1.869 |            0.444 |               24 |            54 |           18 |                           0.574 |
| learned_q                 | selected_best_minus_q_chosen        |             1.213 |                          0.594 |                           1.978 |            0.407 |               22 |            54 |           18 |                           0.574 |
| learned_q                 | selected_best_minus_selected_second |             0.906 |                          0.559 |                           1.299 |            0.593 |               32 |            54 |           18 |                           0.574 |
| permanent_bilateral_3pass | q_chosen_minus_bilateral            |            -0.014 |                         -0.718 |                           0.551 |            0.296 |               16 |            54 |           18 |                           0.611 |
| permanent_bilateral_3pass | selected_best_minus_bilateral       |             1.269 |                          0.827 |                           1.748 |            0.426 |               23 |            54 |           18 |                           0.611 |
| permanent_bilateral_3pass | selected_best_minus_q_chosen        |             1.283 |                          0.695 |                           1.918 |            0.370 |               20 |            54 |           18 |                           0.611 |
| permanent_bilateral_3pass | selected_best_minus_selected_second |             0.989 |                          0.711 |                           1.269 |            0.500 |               27 |            54 |           18 |                           0.611 |
| random_feasible           | q_chosen_minus_bilateral            |            -0.042 |                         -0.337 |                           0.226 |            0.000 |                0 |            54 |           18 |                           0.519 |
| random_feasible           | selected_best_minus_bilateral       |             0.653 |                          0.366 |                           0.947 |            0.000 |                0 |            54 |           18 |                           0.519 |
| random_feasible           | selected_best_minus_q_chosen        |             0.696 |                          0.396 |                           1.093 |            0.019 |                1 |            54 |           18 |                           0.519 |
| random_feasible           | selected_best_minus_selected_second |             0.594 |                          0.348 |                           0.862 |            0.000 |                0 |            54 |           18 |                           0.519 |

## Interpretation

Selection-sample best actions are chosen on the first half of rollouts. All reported action gaps use only the independent evaluation half.