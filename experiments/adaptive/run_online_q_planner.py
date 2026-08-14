"""Train and evaluate the online institutional Q-learning planner.

There are no scheduled institutional baselines in this experiment. The learner
interacts directly with simulated games during training, updates after every
round, and is frozen only for held-out evaluation on disjoint random seeds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.agents import Agent
from engine.policies import (
    AgentPolicy,
    CompetitiveTradePolicy,
    CooperativeTradePolicy,
    FairnessSensitiveTradePolicy,
    HoardingTradePolicy,
    NeedBasedTradePolicy,
    SelfishTradePolicy,
)
from engine.resources import create_random_balanced_access_profiles
from experiments.adaptive.capacity_coordination import (
    CapacityInstitutionEnvironment,
    PolicyBuilder,
)
from experiments.adaptive.online_q_planner import (
    ACTION_ORDER,
    OnlineTabularQPlanner,
    QLearningConfig,
    state_v2_theoretical_size,
    state_v3_theoretical_size,
)

def _string_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}.")
    return value


OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs" / "adaptive_planner"
CSV_DIR = OUTPUT_ROOT / "csv"
MODEL_DIR = OUTPUT_ROOT / "model"

TRAIN_EPISODES = _int_env("TRAIN_EPISODES", 12000, minimum=1)
EVAL_SEEDS = _int_env("EVAL_SEEDS", 100, minimum=1)
ROUNDS = _int_env("ROUNDS", 20, minimum=1)
MAX_BUILDS = _int_env("MAX_BUILDS", 4, minimum=1)
MAX_CAPACITY = _int_env("MAX_CAPACITY", 8, minimum=1)
INITIAL_CAPACITY = _int_env("INITIAL_CAPACITY", MAX_CAPACITY, minimum=0)
CAPACITY_RECOVERY = _int_env("CAPACITY_RECOVERY", 1, minimum=0)
EQUITY_WEIGHT = _float_env("EQUITY_WEIGHT", 0.25, minimum=0.0, maximum=1.0)
TRAIN_RANDOM_SEED = _int_env("TRAIN_RANDOM_SEED", 20260717, minimum=0)
TRAIN_SEED_OFFSET = _int_env("TRAIN_SEED_OFFSET", 100000, minimum=0)
EVAL_SEED_OFFSET = _int_env("EVAL_SEED_OFFSET", 500000, minimum=0)

ALPHA_START = _float_env("ALPHA_START", 0.25, minimum=0.0)
ALPHA_FLOOR = _float_env("ALPHA_FLOOR", 0.01, minimum=0.0)
GAMMA = _float_env("GAMMA", 1.0, minimum=0.0, maximum=1.0)
EPSILON_START = _float_env("EPSILON_START", 1.0, minimum=0.0, maximum=1.0)
EPSILON_END = _float_env("EPSILON_END", 0.03, minimum=0.0, maximum=1.0)
EPSILON_DECAY_FRACTION = _float_env(
    "EPSILON_DECAY_FRACTION",
    0.90,
    minimum=0.0,
    maximum=1.0,
)
STATE_VERSION = _string_env("STATE_VERSION", "v1")
LEARNING_METHOD = _string_env("LEARNING_METHOD", "one_step_q")
N_STEP = _int_env("N_STEP", 1, minimum=1)


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    initial_classes: tuple[type[AgentPolicy], ...]
    change_round: int | None = None
    changed_classes: tuple[type[AgentPolicy], ...] | None = None


MIXED_CLASSES: tuple[type[AgentPolicy], ...] = (
    CooperativeTradePolicy,
    SelfishTradePolicy,
    HoardingTradePolicy,
    CompetitiveTradePolicy,
    FairnessSensitiveTradePolicy,
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="need_based",
        label="Need-based",
        initial_classes=(NeedBasedTradePolicy,) * 5,
    ),
    Scenario(
        name="cooperative",
        label="Cooperative",
        initial_classes=(CooperativeTradePolicy,) * 5,
    ),
    Scenario(
        name="hoarding",
        label="Hoarding",
        initial_classes=(HoardingTradePolicy,) * 5,
    ),
    Scenario(
        name="mixed",
        label="Mixed",
        initial_classes=MIXED_CLASSES,
    ),
    Scenario(
        name="need_based_to_competitive",
        label="Need-based → competitive",
        initial_classes=(NeedBasedTradePolicy,) * 5,
        change_round=10,
        changed_classes=(CompetitiveTradePolicy,) * 5,
    ),
    Scenario(
        name="cooperative_to_hoarding",
        label="Cooperative → hoarding",
        initial_classes=(CooperativeTradePolicy,) * 5,
        change_round=10,
        changed_classes=(HoardingTradePolicy,) * 5,
    ),
)

RNG_STREAM_ACCESS = 0
RNG_STREAM_ASSIGNMENT = 3


def make_rng(seed: int, stream_id: int, *extra: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(stream_id), *map(int, extra)])
    )


def prepare_output_dirs() -> None:
    for path in [CSV_DIR, MODEL_DIR]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def policy_builder_from_classes(
    classes_by_agent: Sequence[type[AgentPolicy]],
) -> PolicyBuilder:
    classes = tuple(classes_by_agent)

    def builder(agents: Sequence[Agent]) -> dict[int, AgentPolicy]:
        if len(agents) != len(classes):
            raise ValueError("Policy-class assignment must match agent count.")
        return {
            agent.id: policy_class()
            for agent, policy_class in zip(agents, classes)
        }

    return builder


def scenario_for_seed(base: Scenario, seed: int) -> Scenario:
    if base.name != "mixed":
        return base
    order = list(MIXED_CLASSES)
    make_rng(seed, RNG_STREAM_ASSIGNMENT).shuffle(order)
    return Scenario(
        name=base.name,
        label=base.label,
        initial_classes=tuple(order),
    )


def create_environment(
    *,
    seed: int,
    scenario: Scenario,
) -> CapacityInstitutionEnvironment:
    profiles = list(
        create_random_balanced_access_profiles(
            make_rng(seed, RNG_STREAM_ACCESS)
        ).values()
    )
    seeded_scenario = scenario_for_seed(scenario, seed)
    initial_builder = policy_builder_from_classes(
        seeded_scenario.initial_classes
    )
    policy_schedule = {}
    if seeded_scenario.change_round is not None:
        if seeded_scenario.changed_classes is None:
            raise ValueError("A change round requires changed policy classes.")
        policy_schedule[seeded_scenario.change_round] = (
            policy_builder_from_classes(seeded_scenario.changed_classes)
        )

    return CapacityInstitutionEnvironment(
        profiles=profiles,
        initial_policy_builder=initial_builder,
        seed=seed,
        total_rounds=ROUNDS,
        max_coordination_capacity=MAX_CAPACITY,
        initial_coordination_capacity=INITIAL_CAPACITY,
        capacity_recovery_per_round=CAPACITY_RECOVERY,
        max_builds_per_agent_per_round=MAX_BUILDS,
        equity_weight=EQUITY_WEIGHT,
        policy_schedule=policy_schedule,
    )


def _episode_summary(
    *,
    env: CapacityInstitutionEnvironment,
    seed: int,
    scenario: Scenario,
    episode: int | None,
    epsilon: float,
    total_reward: float,
    td_errors: list[float],
) -> dict[str, object]:
    final = env.game.history[-1]
    counts = Counter(row["action"] for row in env.action_history)
    return {
        "episode": episode,
        "seed": seed,
        "scenario": scenario.name,
        "scenario_label": scenario.label,
        "change_round": scenario.change_round,
        "epsilon": epsilon,
        "total_reward": total_reward,
        "mean_absolute_td_error": (
            float(np.mean(np.abs(td_errors))) if td_errors else float("nan")
        ),
        "final_total_score": final["total_score"],
        "final_mean_score": final["total_score"] / len(env.game.agents),
        "final_min_score": final["min_score"],
        "bottom_two_mean_score": float(
            np.mean(sorted(agent.score for agent in env.game.agents)[:2])
        ),
        "final_max_score": final["max_score"],
        "final_score_gap": final["score_gap"],
        "final_welfare": final["planner_welfare_after"],
        "final_coordination_capacity": env.coordination_capacity,
        "cumulative_capacity_cost": sum(
            int(row["capacity_realized_cost"])
            for row in env.action_history
        ),
        "cumulative_workload_units": sum(
            int(row["workload_units"])
            for row in env.action_history
        ),
        **{
            f"rounds_{action}": int(counts[action])
            for action in ACTION_ORDER
        },
    }


def train_planner() -> tuple[OnlineTabularQPlanner, pd.DataFrame]:
    planner = OnlineTabularQPlanner(
        config=QLearningConfig(
            gamma=GAMMA,
            alpha_start=ALPHA_START,
            alpha_floor=ALPHA_FLOOR,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_fraction=EPSILON_DECAY_FRACTION,
            learning_method=LEARNING_METHOD,
            n_step=N_STEP,
        ),
        state_version=STATE_VERSION,
    )
    rng = np.random.default_rng(TRAIN_RANDOM_SEED)
    episode_rows: list[dict[str, object]] = []

    for episode in range(TRAIN_EPISODES):
        scenario = SCENARIOS[int(rng.integers(0, len(SCENARIOS)))]
        seed = TRAIN_SEED_OFFSET + episode
        env = create_environment(seed=seed, scenario=scenario)
        epsilon = planner.epsilon_for_episode(episode, TRAIN_EPISODES)
        total_reward = 0.0
        td_errors: list[float] = []
        transitions = []

        while not env.terminated:
            observation = env.observation()
            available_actions = env.available_actions()
            planner.record_training_visit(observation)
            action = planner.choose_action(
                observation=observation,
                available_actions=available_actions,
                rng=rng,
                epsilon=epsilon,
                tie_break="random",
            )
            next_observation, reward, terminated, _ = env.step(action)
            next_available = (
                () if terminated else env.available_actions()
            )
            transition = {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "next_available_actions": next_available,
                "terminated": terminated,
            }
            transitions.append(transition)
            if planner.config.learning_method == "one_step_q":
                td_errors.append(
                    planner.update(
                        observation=observation,
                        action=action,
                        reward=reward,
                        next_observation=next_observation,
                        next_available_actions=next_available,
                        terminated=terminated,
                    )
                )
            total_reward += reward

        if planner.config.learning_method == "n_step_q":
            td_errors.extend(
                planner.update_episode_n_step(
                    transitions,
                    n_step=planner.config.n_step,
                )
            )
        elif planner.config.learning_method == "monte_carlo":
            td_errors.extend(planner.update_episode_monte_carlo(transitions))

        episode_rows.append(
            _episode_summary(
                env=env,
                seed=seed,
                scenario=scenario_for_seed(scenario, seed),
                episode=episode,
                epsilon=epsilon,
                total_reward=total_reward,
                td_errors=td_errors,
            )
        )

        if (episode + 1) % max(1, min(500, TRAIN_EPISODES // 20)) == 0:
            recent = episode_rows[-min(200, len(episode_rows)):]
            print(
                f"training episode {episode + 1}/{TRAIN_EPISODES}; "
                f"epsilon={epsilon:.3f}; "
                f"recent welfare={np.mean([row['final_welfare'] for row in recent]):.2f}; "
                f"states={len(planner.q_table)}"
            )

    return planner, pd.DataFrame(episode_rows)


def evaluate_planner(
    planner: OnlineTabularQPlanner,
    *,
    eval_seeds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    summary_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    trained_states = frozenset(planner.q_table)
    evaluation_unseen_states: set[tuple[int, ...]] = set()

    effective_eval_seeds = EVAL_SEEDS if eval_seeds is None else eval_seeds
    total_games = effective_eval_seeds * len(SCENARIOS)
    completed = 0
    for eval_index in range(effective_eval_seeds):
        seed = EVAL_SEED_OFFSET + eval_index
        for scenario in SCENARIOS:
            seeded_scenario = scenario_for_seed(scenario, seed)
            env = create_environment(seed=seed, scenario=scenario)
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, 91])
            )
            total_reward = 0.0

            while not env.terminated:
                observation = env.observation()
                available_actions = env.available_actions()
                state = planner.state_key(observation)
                if state not in trained_states:
                    evaluation_unseen_states.add(state)
                diagnostics = planner.diagnostics(
                    observation,
                    available_actions,
                )
                action = planner.choose_action(
                    observation=observation,
                    available_actions=available_actions,
                    rng=rng,
                    epsilon=0.0,
                )
                _, reward, _, _ = env.step(action)
                total_reward += reward
                env.game.history[-1].update(diagnostics)

            summary_rows.append(
                _episode_summary(
                    env=env,
                    seed=seed,
                    scenario=seeded_scenario,
                    episode=None,
                    epsilon=0.0,
                    total_reward=total_reward,
                    td_errors=[],
                )
            )

            for row in env.game.history:
                round_rows.append(
                    {
                        "seed": seed,
                        "scenario": seeded_scenario.name,
                        "scenario_label": seeded_scenario.label,
                        "change_round": seeded_scenario.change_round,
                        **row,
                    }
                )

            for agent in env.game.agents:
                agent_rows.append(
                    {
                        "seed": seed,
                        "scenario": seeded_scenario.name,
                        "scenario_label": seeded_scenario.label,
                        "agent_id": agent.id,
                        "policy_class_name": (
                            env.game.policies[agent.id].__class__.__name__
                        ),
                        "score": agent.score,
                        "infrastructure": agent.infrastructure,
                        "production_sites": agent.production_sites,
                        "advanced_sites": agent.advanced_sites,
                        "innovation": agent.innovation,
                    }
                )

            completed += 1
        if (eval_index + 1) % max(1, min(10, effective_eval_seeds)) == 0:
            print(f"evaluation completed {completed}/{total_games} games")

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(round_rows),
        pd.DataFrame(agent_rows),
        len(evaluation_unseen_states),
    )


def build_run_metadata(
    *,
    planner: OnlineTabularQPlanner,
    trained_q_states: int,
    evaluation_unseen_states: int,
) -> dict[str, object]:
    """Return persisted run metadata for a completed planner run."""
    return {
        "train_episodes": TRAIN_EPISODES,
        "eval_seeds": EVAL_SEEDS,
        "rounds": ROUNDS,
        "max_builds": MAX_BUILDS,
        "max_capacity": MAX_CAPACITY,
        "initial_capacity": INITIAL_CAPACITY,
        "capacity_recovery": CAPACITY_RECOVERY,
        "equity_weight": EQUITY_WEIGHT,
        "train_random_seed": TRAIN_RANDOM_SEED,
        "train_seed_offset": TRAIN_SEED_OFFSET,
        "eval_seed_offset": EVAL_SEED_OFFSET,
        "trained_q_states": trained_q_states,
        "evaluation_unseen_states": evaluation_unseen_states,
        "q_states": trained_q_states,
        "q_config": asdict(planner.config),
        "state_version": planner.state_version,
        "learning_method": planner.config.learning_method,
        "n_step": planner.config.n_step,
        "state_v2_theoretical_size": (
            state_v2_theoretical_size()
            if planner.state_version == "v2"
            else None
        ),
        "state_v3_theoretical_size": (
            state_v3_theoretical_size()
            if planner.state_version == "v3"
            else None
        ),
        "actions": list(ACTION_ORDER),
    }


def main() -> None:
    if INITIAL_CAPACITY > MAX_CAPACITY:
        raise ValueError(
            "INITIAL_CAPACITY cannot exceed MAX_CAPACITY "
            f"({INITIAL_CAPACITY} > {MAX_CAPACITY})."
        )

    prepare_output_dirs()
    print("Online institutional Q planner")
    print(f"TRAIN_EPISODES: {TRAIN_EPISODES}")
    print(f"EVAL_SEEDS: {EVAL_SEEDS}")
    print(f"ROUNDS: {ROUNDS}")
    print(
        "Capacity: "
        f"initial={INITIAL_CAPACITY}, max={MAX_CAPACITY}, "
        f"recovery={CAPACITY_RECOVERY}/round"
    )
    print(f"EQUITY_WEIGHT: {EQUITY_WEIGHT}")
    print(f"Actions: {ACTION_ORDER}")
    print(f"Output: {OUTPUT_ROOT}")

    planner, training = train_planner()
    trained_q_states = len(planner.q_table)
    planner.save(MODEL_DIR / "online_q_table.json")
    training.to_csv(CSV_DIR / "training_episodes.csv", index=False)

    summary, history, agents, evaluation_unseen_states = evaluate_planner(planner)
    summary.to_csv(CSV_DIR / "evaluation_summary_by_seed.csv", index=False)
    history.to_csv(CSV_DIR / "evaluation_round_history.csv", index=False)
    agents.to_csv(CSV_DIR / "evaluation_final_agents.csv", index=False)

    metadata = build_run_metadata(
        planner=planner,
        trained_q_states=trained_q_states,
        evaluation_unseen_states=evaluation_unseen_states,
    )
    pd.DataFrame([metadata]).to_json(
        CSV_DIR / "run_metadata.json",
        orient="records",
        indent=2,
    )

    from experiments.analysis.analyze_online_q_planner import (
        regenerate_online_planner_outputs,
    )

    regenerate_online_planner_outputs(output_root=OUTPUT_ROOT)


if __name__ == "__main__":
    main()
