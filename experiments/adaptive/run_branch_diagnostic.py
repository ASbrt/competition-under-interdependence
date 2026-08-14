from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adaptive.baseline_policies import (  # noqa: E402
    LearnedQPolicy,
    PermanentActionPolicy,
    PlannerPolicy,
    UniformRandomFeasiblePolicy,
)
from experiments.adaptive.evaluate_online_planner_baselines import gini  # noqa: E402
from experiments.adaptive.online_q_planner import OnlineTabularQPlanner  # noqa: E402
from experiments.adaptive.run_online_q_planner import SCENARIOS, create_environment, scenario_for_seed  # noqa: E402


OUT = REPO_ROOT / "experiments" / "outputs" / "branch_diagnostic"
MODEL = REPO_ROOT / "experiments" / "outputs" / "adaptive_planner" / "model" / "online_q_table.json"
CHECKPOINT_ROUNDS = (3, 10, 16)
BASE_SEED_START = 700_000
BASE_GAMES_PER_SCENARIO = 3
REPLICATIONS_PER_ACTION = 40
BOOTSTRAP_DRAWS = 10_000


def final_metrics(env) -> dict[str, float]:
    final = env.game.history[-1]
    scores = [agent.score for agent in env.game.agents]
    return {
        "final_total_score": float(final["total_score"]),
        "final_mean_score": float(final["total_score"]) / len(scores),
        "final_min_score": float(final["min_score"]),
        "bottom_two_mean_score": float(np.mean(sorted(scores)[:2])),
        "final_score_gap": float(final["score_gap"]),
        "gini_score": gini(scores),
        "final_welfare": float(final["planner_welfare_after"]),
        "cumulative_capacity_cost": float(
            sum(int(row["capacity_realized_cost"]) for row in env.action_history)
        ),
    }


def policy_rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([20260811, *map(int, parts)]))


def continue_episode(
    *,
    env,
    policy: PlannerPolicy,
    rng: np.random.Generator,
    seed: int,
    scenario: str,
) -> None:
    policy.reset_game(seed=seed, scenario=scenario)
    while not env.terminated:
        action = policy.choose_action(env.observation(), env.available_actions(), rng)
        env.step(action)


def branch_base_games() -> list[tuple[int, object]]:
    games = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for offset in range(BASE_GAMES_PER_SCENARIO):
            games.append((BASE_SEED_START + scenario_index * 100 + offset, scenario))
    return games


def collect_checkpoints(planner: OnlineTabularQPlanner) -> pd.DataFrame:
    learned = LearnedQPolicy(planner)
    rows = []
    state_id = 0
    for base_game_id, (seed, scenario) in enumerate(branch_base_games(), start=1):
        seeded = scenario_for_seed(scenario, seed)
        env = create_environment(seed=seed, scenario=scenario)
        while not env.terminated:
            observation = env.observation()
            q_chosen_action = learned.choose_action(observation, env.available_actions(), policy_rng(seed),)
            if observation.round_number in CHECKPOINT_ROUNDS:
                state_id += 1
                rows.append(
                    {
                        "branch_state_id": state_id,
                        "base_game_id": base_game_id,
                        "seed": seed,
                        "scenario": seeded.name,
                        "scenario_label": seeded.label,
                        "round": observation.round_number,
                        "q_chosen_action": q_chosen_action,
                        "feasible_actions": "|".join(env.available_actions()),
                        "n_feasible_actions": len(env.available_actions()),
                        "env": copy.deepcopy(env),
                    }
                )
            env.step(q_chosen_action)
    return pd.DataFrame(rows)


def run_rollouts(planner: OnlineTabularQPlanner, checkpoints: pd.DataFrame) -> pd.DataFrame:
    policies: list[PlannerPolicy] = [
        LearnedQPolicy(planner),
        UniformRandomFeasiblePolicy(),
        PermanentActionPolicy(),
    ]
    rows = []
    total = len(checkpoints)
    for index, cp in enumerate(checkpoints.itertuples(), start=1):
        print(f"rollouts {index}/{total}: branch_state_id={cp.branch_state_id}")
        actions = cp.feasible_actions.split("|")
        for action in actions:
            for policy_index, policy in enumerate(policies):
                deterministic_metrics = None
                for replication in range(REPLICATIONS_PER_ACTION):
                    if deterministic_metrics is None:
                        branch = copy.deepcopy(cp.env)
                        branch.step(action)
                        continue_episode(
                            env=branch,
                            policy=policy,
                            rng=policy_rng(cp.branch_state_id, policy_index, replication),
                            seed=cp.seed,
                            scenario=cp.scenario,
                        )
                        deterministic_metrics = final_metrics(branch)
                    metrics = deterministic_metrics
                    if isinstance(policy, UniformRandomFeasiblePolicy):
                        deterministic_metrics = None
                    split = "selection" if replication < REPLICATIONS_PER_ACTION // 2 else "evaluation"
                    rows.append(
                        {
                            "branch_state_id": cp.branch_state_id,
                            "base_game_id": cp.base_game_id,
                            "seed": cp.seed,
                            "scenario": cp.scenario,
                            "round": cp.round,
                            "first_action": action,
                            "q_chosen_action": cp.q_chosen_action,
                            "continuation_policy": policy.name,
                            "replication": replication,
                            "split": split,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) <= 1:
        mean = float(np.mean(values)) if len(values) else float("nan")
        return mean, mean
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def summarize_gaps(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, policy_name), data in values.groupby(["branch_state_id", "continuation_policy"], observed=False):
        selection = (
            data[data["split"] == "selection"]
            .groupby("first_action", observed=False)["final_welfare"]
            .mean()
            .sort_values(ascending=False)
        )
        best = selection.index[0]
        second = selection.index[1] if len(selection) > 1 else selection.index[0]
        q_chosen = data["q_chosen_action"].iloc[0]
        eval_data = data[data["split"] == "evaluation"]
        for comparison, left, right in [
            ("selected_best_minus_selected_second", best, second),
            ("selected_best_minus_q_chosen", best, q_chosen),
            ("selected_best_minus_bilateral", best, "bilateral_3pass"),
            ("q_chosen_minus_bilateral", q_chosen, "bilateral_3pass"),
        ]:
            if left not in set(eval_data["first_action"]) or right not in set(eval_data["first_action"]):
                continue
            left_values = eval_data[eval_data["first_action"] == left].sort_values("replication")
            right_values = eval_data[eval_data["first_action"] == right].sort_values("replication")
            merged = left_values[["replication", "final_welfare"]].merge(
                right_values[["replication", "final_welfare"]],
                on="replication",
                suffixes=("_left", "_right"),
            )
            diffs = merged["final_welfare_left"].to_numpy(float) - merged["final_welfare_right"].to_numpy(float)
            low, high = ci(diffs)
            first = data.iloc[0]
            rows.append(
                {
                    "branch_state_id": state_id,
                    "base_game_id": int(first["base_game_id"]),
                    "seed": int(first["seed"]),
                    "scenario": first["scenario"],
                    "round": int(first["round"]),
                    "continuation_policy": policy_name,
                    "comparison": comparison,
                    "selection_best_action": best,
                    "selection_second_action": second,
                    "q_chosen_action": q_chosen,
                    "mean_difference": float(np.mean(diffs)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci_excludes_zero": int(low > 0 or high < 0),
                    "paired_eval_replications": int(len(diffs)),
                    "q_equals_selection_best": int(q_chosen == best),
                }
            )
    return pd.DataFrame(rows)


def cluster_ci(rows: pd.DataFrame, column: str) -> tuple[float, float]:
    cluster_stats = [
        (float(group[column].sum()), len(group))
        for _, group in rows.groupby("base_game_id", observed=False)
    ]
    sums = np.asarray([item[0] for item in cluster_stats], dtype=np.float64)
    counts = np.asarray([item[1] for item in cluster_stats], dtype=np.float64)
    rng = np.random.default_rng(20260812)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(sums), size=len(sums))
        draws[_] = float(sums[sampled].sum() / counts[sampled].sum())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def aggregate_summary(gaps: pd.DataFrame, checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy_name, comparison), data in gaps.groupby(["continuation_policy", "comparison"], observed=False):
        low, high = cluster_ci(data, "mean_difference")
        rows.append(
            {
                "continuation_policy": policy_name,
                "comparison": comparison,
                "mean_difference": float(data["mean_difference"].mean()),
                "base_game_clustered_ci95_low": low,
                "base_game_clustered_ci95_high": high,
                "resolved_share": float(data["ci_excludes_zero"].mean()),
                "resolved_count": int(data["ci_excludes_zero"].sum()),
                "checkpoints": int(data["branch_state_id"].nunique()),
                "base_games": int(data["base_game_id"].nunique()),
                "q_equals_selection_best_share": float(data["q_equals_selection_best"].mean()),
            }
        )
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame) -> str:
    q_best = summary[
        (summary["continuation_policy"] == "learned_q")
        & (summary["comparison"] == "selected_best_minus_q_chosen")
    ].iloc[0]
    rf_best = summary[
        (summary["continuation_policy"] == "random_feasible")
        & (summary["comparison"] == "selected_best_minus_q_chosen")
    ].iloc[0]
    if q_best["resolved_share"] >= 0.33 and rf_best["resolved_share"] >= 0.25:
        return "SUPPORTIVE"
    if q_best["mean_difference"] > 0 or rf_best["mean_difference"] > 0:
        return "MIXED / INCONCLUSIVE"
    return "UNSUPPORTIVE"


def write_report(checkpoints: pd.DataFrame, gaps: pd.DataFrame, summary: pd.DataFrame) -> None:
    classification = verdict(summary)
    lines = [
        "# Counterfactual branch diagnostic",
        "",
        f"Verdict: **{classification}**.",
        "",
        "## Design",
        "",
        f"- Base games: {checkpoints['base_game_id'].nunique()} using seeds {', '.join(map(str, sorted(checkpoints['seed'].unique())))}.",
        f"- Checkpoints: {len(checkpoints)} at rounds {CHECKPOINT_ROUNDS}.",
        f"- Continuations per feasible action: {REPLICATIONS_PER_ACTION}.",
        f"- Selection/evaluation split: {REPLICATIONS_PER_ACTION // 2} / {REPLICATIONS_PER_ACTION // 2}.",
        "- Candidate branches are deep-copied from the same checkpoint. For a checkpoint, continuation policy, and replication index, every feasible first action uses the same production RNG state at the branch point and the same policy RNG seed pattern.",
        "- Branches are not forced to stay identical after actions diverge; switching costs, workload, future capacity, and later feasibility are recalculated normally.",
        "",
        "## Aggregate gaps",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "Selection-sample best actions are chosen on the first half of rollouts. All reported action gaps use only the independent evaluation half.",
    ]
    (OUT / "branch_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    planner = OnlineTabularQPlanner.load(MODEL)
    checkpoints = collect_checkpoints(planner)
    values = run_rollouts(planner, checkpoints)
    public_checkpoints = checkpoints.drop(columns=["env"])
    gaps = summarize_gaps(values)
    summary = aggregate_summary(gaps, public_checkpoints)
    public_checkpoints.to_csv(OUT / "branch_checkpoints.csv", index=False)
    values.to_csv(OUT / "branch_rollout_values.csv", index=False)
    gaps.to_csv(OUT / "branch_evaluation_gaps.csv", index=False)
    summary.to_csv(OUT / "branch_aggregate_summary.csv", index=False)
    metadata = {
        "model": str(MODEL.relative_to(REPO_ROOT)),
        "base_seed_start": BASE_SEED_START,
        "base_games_per_scenario": BASE_GAMES_PER_SCENARIO,
        "checkpoint_rounds": CHECKPOINT_ROUNDS,
        "replications_per_action": REPLICATIONS_PER_ACTION,
        "selection_replications": REPLICATIONS_PER_ACTION // 2,
        "evaluation_replications": REPLICATIONS_PER_ACTION // 2,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "verdict": verdict(summary),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(public_checkpoints, gaps, summary)
    print(f"wrote branch diagnostic to {OUT}")


if __name__ == "__main__":
    main()
