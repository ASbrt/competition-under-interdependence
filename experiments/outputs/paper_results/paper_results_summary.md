# Paper result tables

These tables are generated from the stored experiment outputs using the uncertainty procedures described in the paper.

## Adaptive policy comparisons

| target_policy             | baseline_policy           | metric        |   mean_difference |   seed_clustered_ci95_low |   seed_clustered_ci95_high |   seed_clusters |   observations |
|:--------------------------|:--------------------------|:--------------|------------------:|--------------------------:|---------------------------:|----------------:|---------------:|
| learned_q                 | permanent_bilateral_3pass | final_welfare |             0.765 |                     0.303 |                      1.218 |             100 |            600 |
| learned_q                 | random_feasible           | final_welfare |             0.136 |                    -0.336 |                      0.608 |             100 |            600 |
| learned_q                 | frequency_informed_random | final_welfare |            -0.133 |                    -0.597 |                      0.324 |             100 |            600 |
| learned_q                 | shuffled_learned_sequence | final_welfare |             0.138 |                    -0.295 |                      0.555 |             100 |            600 |
| frequency_informed_random | permanent_bilateral_3pass | final_welfare |             0.897 |                     0.469 |                      1.315 |             100 |            600 |
| frequency_informed_random | random_feasible           | final_welfare |             0.269 |                    -0.185 |                      0.720 |             100 |            600 |

## Behavioral-composition endpoint comparisons

| sweep_label              | condition_label    |   n_matched_seeds |   mean_change_5_minus_0 |   ci95_low |   ci95_high |
|:-------------------------|:-------------------|------------------:|------------------------:|-----------:|------------:|
| Cooperative → hoarding   | Bilateral 3-pass   |               100 |                 -28.920 |    -34.700 |     -23.210 |
| Cooperative → hoarding   | Central capped (2) |               100 |                   6.710 |      2.990 |      10.470 |
| Cooperative → hoarding   | Central clearing   |               100 |                  17.120 |     12.220 |      22.100 |
| Cooperative → hoarding   | No trade           |               100 |                  -0.010 |     -0.030 |       0.000 |
| Cooperative → hoarding   | Public pool        |               100 |                -110.580 |   -114.160 |    -107.090 |
| Need-based → competitive | Bilateral 3-pass   |               100 |                 -41.390 |    -46.990 |     -35.770 |
| Need-based → competitive | Central capped (2) |               100 |                  -5.100 |     -9.060 |      -1.000 |
| Need-based → competitive | Central clearing   |               100 |                 -18.000 |    -22.840 |     -13.140 |
| Need-based → competitive | No trade           |               100 |                   0.010 |      0.000 |       0.030 |
| Need-based → competitive | Public pool        |               100 |                -116.510 |   -119.270 |    -113.760 |

## Central versus equity-oriented central coordination

| metric            | metric_definition         | comparison                            |   mean_difference |   seed_clustered_ci95_low |   seed_clustered_ci95_high |   seed_clusters |   observations |
|:------------------|:--------------------------|:--------------------------------------|------------------:|--------------------------:|---------------------------:|----------------:|---------------:|
| final_total_score | final total score         | central_clearing_minus_equity_central |             3.286 |                     1.861 |                      4.704 |             200 |           1400 |
| final_score_gap   | final max-min score range | central_clearing_minus_equity_central |             2.356 |                     1.868 |                      2.844 |             200 |           1400 |

## Counterfactual branch diagnostic

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