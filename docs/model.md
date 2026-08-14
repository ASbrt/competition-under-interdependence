# Model

The model is a stylized agent-based exchange economy built to study institutional
coordination under resource interdependence. It separates the resource economy,
agent decision rules, and exchange institutions so that behavioral populations
and institutional arrangements can be varied independently.

## Agents and resources

Each game contains five agents and five resource types:

- materials,
- components,
- food,
- energy,
- knowledge.

Agents start with asymmetric resource-access profiles. Development projects
require complementary bundles, so an agent can have substantial resources and
still be blocked by one missing input. Exchange can therefore change both
current development and future productive capacity.

## Round sequence

A game lasts 20 rounds. Each round contains four broad stages:

1. resource production,
2. institution-mediated exchange,
3. agent building,
4. score, capacity, and access updates.

Agents can build infrastructure, production sites, advanced production, and
innovation. A successful build can change later production or development
opportunities, which makes exchange outcomes path dependent.

## Behavioral policies

The main fixed experiment uses seven population conditions:

- need-based,
- cooperative,
- selfish,
- hoarding,
- competitive,
- fairness-sensitive,
- mixed.

The first six are homogeneous populations. The mixed population contains one
cooperative, selfish, hoarding, competitive, and fairness-sensitive agent.

All policy classes use the same underlying decision architecture and feasible
project set. They differ in utility weights and in exchange parameters governing
requests, offers, acceptance, reserves, payment limits, and public-pool
contributions. These are stylized behavioral policy packages rather than
estimated psychological types.

The parameter definitions are implemented in `engine/policies.py`.

## Exchange institutions

The fixed benchmark includes eleven exchange conditions. The adaptive planner
chooses among seven of them:

| Adaptive action | Role |
| --- | --- |
| `bilateral_3pass` | repeated direct bilateral exchange |
| `clearinghouse` | aggregates compatible exchange opportunities |
| `public_pool` | round-local contribution and allocation pool |
| `subsidized_catch_up` | transfers resources toward bottlenecked agents |
| `central_cap2` | central greedy matching with a trade cap |
| `equity_cap2` | capped central matching with an equity priority |
| `central_full` | unrestricted greedy central matching |

The central mechanisms are not globally optimal market-clearing mechanisms.
They are greedy resource matchers implemented in `engine/institutions.py`.
Because the institutions change several operational features at once, the paper
interprets their effects as differences between complete protocol packages.

## Build modes

The fixed benchmark crosses the behavioral populations and institutions with two
build modes:

- `development_oriented`,
- `crown_aware`.

Both use the same utility-based build system. The crown-aware mode adds a term
related to infrastructure- and innovation-leader bonuses. The main composition
and central-versus-equity analyses use the development-oriented condition.

## Coordination capacity

The adaptive environment makes stronger coordination scarce. Let `G_t` denote
available coordination capacity. The paper specification uses:

```text
maximum capacity      G_max = 8
initial capacity            = 8
recovery per round      g   = 1
```

An institution can have a base cost, a switching cost, and a workload cost. The
planner must be able to cover the institution's maximum possible commitment
before choosing it. After execution, only the realized cost is deducted. The
specific cost table is defined in `experiments/adaptive/capacity_coordination.py`.

These values are modeling assumptions used to create an intertemporal scarcity
problem. They are not estimates of administrative or political capacity.

## Planner welfare and reward

Planner welfare is

```text
Phi_t = (1 - lambda) * mean_score_t + lambda * min_score_t
```

with `lambda = 0.25` in the reported adaptive experiment. The reward is the
round-to-round change in `Phi`. With discount factor `gamma = 1`, summing rewards
over the game is equivalent to final planner welfare because all agents begin at
zero score.

## Q-learning state

The reported planner uses a seven-feature discretized state:

1. game phase,
2. remaining coordination capacity,
3. score-gap category,
4. bottleneck severity,
5. idle productive infrastructure,
6. viability of voluntary exchange,
7. previous institution.

The planner does not observe full inventories, targets, behavioral identities,
or every possible exchange opportunity. Behavioral composition is therefore
hidden except through its consequences for these observed features.

The exact bins and the Q-learning update are implemented in
`experiments/adaptive/online_q_planner.py`. The reported training specification
uses 12,000 episodes, `alpha_start = 0.25`, `alpha_floor = 0.01`,
`epsilon_start = 1.0`, `epsilon_end = 0.03`, an exploration decay fraction of
`0.90`, and `gamma = 1.0`.

## Evaluation design

Training uses simulation seeds `100000` through `111999`. Evaluation uses seeds
`500000` through `500099`, so evaluation seeds are disjoint from training. The
same evaluation set was subsequently used for additional learner variants; the
paper therefore treats it as held out from training but not as an untouched
model-selection test set.
