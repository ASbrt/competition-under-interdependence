# Online institutional planner results

The planner was trained online through direct interaction with simulated games. There are no scheduled institutional baselines in this experiment. Governance capacity is part of the environment state: coordinated institutions deplete it according to base, switching, and realized workload costs, while unused capacity regenerates between rounds.

## Held-out outcomes

- Need-based: total score 136.51; weakest-agent score 16.12; welfare 24.51; mean capacity cost 24.51; most-used institution Bilateral 3-pass.
- Cooperative: total score 120.52; weakest-agent score 12.23; welfare 21.14; mean capacity cost 24.38; most-used institution Bilateral 3-pass.
- Hoarding: total score 101.98; weakest-agent score 9.94; welfare 17.78; mean capacity cost 25.27; most-used institution Bilateral 3-pass.
- Mixed: total score 110.15; weakest-agent score 10.78; welfare 19.22; mean capacity cost 24.65; most-used institution Bilateral 3-pass.
- Need-based → competitive: total score 118.19; weakest-agent score 12.31; welfare 20.81; mean capacity cost 24.81; most-used institution Bilateral 3-pass.
- Cooperative → hoarding: total score 102.72; weakest-agent score 8.94; welfare 17.64; mean capacity cost 25.09; most-used institution Bilateral 3-pass.

## Institutional use and cost

- Bilateral 3-pass: selected 6951 held-out rounds; mean realized capacity cost 0.00; mean workload 8.25 resource-handling units; mean welfare reward 1.162.
- Subsidized catch-up: selected 2265 held-out rounds; mean realized capacity cost 2.58; mean workload 0.90 resource-handling units; mean welfare reward 0.817.
- Clearinghouse: selected 1544 held-out rounds; mean realized capacity cost 2.76; mean workload 5.49 resource-handling units; mean welfare reward 0.988.
- Round-local public pool: selected 565 held-out rounds; mean realized capacity cost 3.34; mean workload 12.25 resource-handling units; mean welfare reward 0.388.
- Central clearing: selected 310 held-out rounds; mean realized capacity cost 4.74; mean workload 2.88 resource-handling units; mean welfare reward 0.694.
- Central capped (2): selected 191 held-out rounds; mean realized capacity cost 3.86; mean workload 3.08 resource-handling units; mean welfare reward 0.662.
- Equity central capped (2): selected 174 held-out rounds; mean realized capacity cost 3.86; mean workload 3.03 resource-handling units; mean welfare reward 0.565.

## Interpretation

The main result is whether the learned institutional mix changes coherently with behavioral composition, bottlenecks, inequality, and available governance capacity. The fixed-institution benchmark remains the external reference for how the individual institutions perform when selected continuously; it is not repeated as a scheduled switching experiment here.