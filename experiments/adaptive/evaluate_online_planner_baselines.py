"""Evaluate the learned Q policy against paired institutional baselines.

All policies are evaluated on the same seed-scenario games. Pooled uncertainty
is clustered by simulation seed so the six scenarios belonging to a seed remain
together during resampling.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adaptive.baseline_policies import (
    FrequencyMatchedRandomPolicy,
    LearnedQPolicy,
    PermanentActionPolicy,
    PlannerPolicy,
    ShuffledLearnedTimingPolicy,
    UniformRandomFeasiblePolicy,
)
from experiments.adaptive.online_q_planner import ACTION_ORDER, OnlineTabularQPlanner
from experiments.adaptive.run_online_q_planner import (
    EVAL_SEED_OFFSET,
    SCENARIOS,
    _episode_summary,
    create_environment,
    scenario_for_seed,
)

DEFAULT_MODEL = REPO_ROOT / "experiments" / "outputs" / "adaptive_planner" / "model" / "online_q_table.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "outputs" / "adaptive_baselines"
METRICS = (
    "final_total_score",
    "final_mean_score",
    "final_min_score",
    "bottom_two_mean_score",
    "final_score_gap",
    "gini_score",
    "final_welfare",
    "cumulative_capacity_cost",
)


def gini(values: Sequence[float]) -> float:
    data = np.sort(np.asarray(values, dtype=np.float64))
    if len(data) == 0 or data.sum() <= 0:
        return 0.0
    index = np.arange(1, len(data) + 1, dtype=np.float64)
    return float((2 * np.sum(index * data)) / (len(data) * data.sum()) - (len(data) + 1) / len(data))


def prepare_output_dirs(root: Path) -> tuple[Path, Path, Path]:
    csv_dir = root / "csv"
    tables_dir = root / "tables"
    markdown_dir = root / "markdown"
    for path in (csv_dir, tables_dir, markdown_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return csv_dir, tables_dir, markdown_dir


def policy_rng(seed: int, policy_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), 271, policy_index]))


def evaluate_policy(
    *,
    policy: PlannerPolicy,
    policy_index: int,
    eval_seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []

    for eval_index in range(eval_seeds):
        seed = EVAL_SEED_OFFSET + eval_index
        for scenario in SCENARIOS:
            seeded_scenario = scenario_for_seed(scenario, seed)
            env = create_environment(seed=seed, scenario=scenario)
            rng = policy_rng(seed, policy_index)
            policy.reset_game(seed=seed, scenario=seeded_scenario.name)
            total_reward = 0.0

            while not env.terminated:
                observation = env.observation()
                diagnostics = (
                    policy.planner.diagnostics(observation, env.available_actions())
                    if isinstance(policy, LearnedQPolicy)
                    else {}
                )
                action = policy.choose_action(
                    observation=observation,
                    available_actions=env.available_actions(),
                    rng=rng,
                )
                _, reward, _, _ = env.step(action)
                if diagnostics:
                    env.game.history[-1].update(diagnostics)
                total_reward += reward

            scores = [agent.score for agent in env.game.agents]
            row = _episode_summary(
                env=env,
                seed=seed,
                scenario=seeded_scenario,
                episode=None,
                epsilon=0.0,
                total_reward=total_reward,
                td_errors=[],
            )
            row.update(
                {
                    "policy_name": policy.name,
                    "policy_label": policy.label,
                    "bottom_two_mean_score": float(np.mean(sorted(scores)[:2])),
                    "gini_score": gini(scores),
                }
            )
            summary_rows.append(row)

            for history_row in env.game.history:
                history_rows.append(
                    {
                        "policy_name": policy.name,
                        "policy_label": policy.label,
                        "seed": seed,
                        "scenario": seeded_scenario.name,
                        "scenario_label": seeded_scenario.label,
                        "change_round": seeded_scenario.change_round,
                        **history_row,
                    }
                )

            for agent in env.game.agents:
                agent_rows.append(
                    {
                        "policy_name": policy.name,
                        "policy_label": policy.label,
                        "seed": seed,
                        "scenario": seeded_scenario.name,
                        "scenario_label": seeded_scenario.label,
                        "agent_id": agent.id,
                        "policy_class_name": env.game.policies[agent.id].__class__.__name__,
                        "score": agent.score,
                        "infrastructure": agent.infrastructure,
                        "production_sites": agent.production_sites,
                        "advanced_sites": agent.advanced_sites,
                        "innovation": agent.innovation,
                    }
                )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(history_rows),
        pd.DataFrame(agent_rows),
    )


def learned_sequences(history: pd.DataFrame) -> dict[tuple[int, str], tuple[str, ...]]:
    sequences = {}
    for key, rows in history.sort_values("round").groupby(["seed", "scenario"]):
        sequences[key] = tuple(rows["planner_action"])
    return sequences


def summarize_policy_outcomes(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["policy_name", "policy_label", "scenario", "scenario_label"], observed=False)
        .agg(
            n=("seed", "nunique"),
            **{f"mean_{metric}": (metric, "mean") for metric in METRICS},
            **{f"sd_{metric}": (metric, "std") for metric in METRICS},
        )
        .reset_index()
    )


def action_allocation(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["policy_name", "policy_label", "scenario", "scenario_label"], observed=False)[
            [f"rounds_{action}" for action in ACTION_ORDER]
        ]
        .mean()
        .reset_index()
        .rename(columns={f"rounds_{action}": action for action in ACTION_ORDER})
    )


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def clustered_seed_interval(
    paired_rows: pd.DataFrame,
    *,
    target_column: str,
    baseline_column: str,
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap paired differences by resampling seed clusters."""
    cluster_stats = []
    for _, rows in paired_rows.groupby("seed", observed=False):
        diff = (
            rows[target_column].to_numpy(dtype=np.float64)
            - rows[baseline_column].to_numpy(dtype=np.float64)
        )
        cluster_stats.append((float(diff.sum()), len(diff)))
    sums = np.asarray([item[0] for item in cluster_stats], dtype=np.float64)
    counts = np.asarray([item[1] for item in cluster_stats], dtype=np.float64)
    draws = np.empty(bootstrap_draws, dtype=np.float64)
    for _ in range(bootstrap_draws):
        sampled = rng.integers(0, len(sums), size=len(sums))
        draws[_] = float(sums[sampled].sum() / counts[sampled].sum())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_comparisons(
    summary: pd.DataFrame,
    *,
    bootstrap_draws: int,
    cluster_bootstrap_draws: int,
    target_policy: str = "learned_q",
) -> pd.DataFrame:
    """Compare the learned policy with each baseline on matched games."""
    target = summary[summary["policy_name"] == target_policy]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260718)
    cluster_rng = np.random.default_rng(20260813)

    # The pooled final-welfare intervals are the intervals reported in the paper.
    # They are generated in a fixed comparison order with one seed-cluster bootstrap.
    paper_cluster_rng = np.random.default_rng(20260811)
    paper_cluster_intervals: dict[str, tuple[float, float]] = {}
    paper_order = [
        "permanent_bilateral_3pass",
        "random_feasible",
        "frequency_informed_random",
        "shuffled_learned_sequence",
    ]
    for baseline_name in paper_order:
        baseline_rows = summary[summary["policy_name"] == baseline_name]
        if baseline_rows.empty:
            continue
        paired = target.merge(
            baseline_rows,
            on=["seed", "scenario"],
            suffixes=("_target", "_baseline"),
        )
        paper_cluster_intervals[baseline_name] = clustered_seed_interval(
            paired,
            target_column="final_welfare_target",
            baseline_column="final_welfare_baseline",
            bootstrap_draws=cluster_bootstrap_draws,
            rng=paper_cluster_rng,
        )

    for policy_name in sorted(set(summary["policy_name"]) - {target_policy}):
        other = summary[summary["policy_name"] == policy_name]
        merged = target.merge(
            other,
            on=["seed", "scenario"],
            suffixes=("_target", "_baseline"),
        )
        for scenario in ["ALL", *[scenario.name for scenario in SCENARIOS]]:
            subset = merged if scenario == "ALL" else merged[merged["scenario"] == scenario]
            if subset.empty:
                continue
            for metric in METRICS:
                diff = (
                    subset[f"{metric}_target"].to_numpy(dtype=np.float64)
                    - subset[f"{metric}_baseline"].to_numpy(dtype=np.float64)
                )
                low, high = bootstrap_interval(diff, rng, bootstrap_draws)
                if scenario == "ALL":
                    if metric == "final_welfare" and policy_name in paper_cluster_intervals:
                        clustered_low, clustered_high = paper_cluster_intervals[policy_name]
                    else:
                        clustered_low, clustered_high = clustered_seed_interval(
                            subset,
                            target_column=f"{metric}_target",
                            baseline_column=f"{metric}_baseline",
                            bootstrap_draws=cluster_bootstrap_draws,
                            rng=cluster_rng,
                        )
                else:
                    clustered_low, clustered_high = low, high
                rows.append(
                    {
                        "baseline_policy": policy_name,
                        "scenario": scenario,
                        "metric": metric,
                        "n_pairs": len(diff),
                        "target_policy": target_policy,
                        "mean_learned_minus_baseline": float(np.mean(diff)),
                        "mean_target_minus_baseline": float(np.mean(diff)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "seed_clustered_ci95_low": clustered_low,
                        "seed_clustered_ci95_high": clustered_high,
                        "share_target_higher": float(np.mean(diff > 0)),
                        "share_learned_higher": float(np.mean(diff > 0)),
                    }
                )
    return pd.DataFrame(rows)


def q_state_diagnostics(
    *,
    planner: OnlineTabularQPlanner,
    learned_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    update_rows = []
    for state, counts in planner.update_counts.items():
        visits = int(planner.visit_counts.get(state, 0))
        for action, count in zip(planner.action_order, counts):
            update_rows.append(
                {
                    "q_state_key": ",".join(map(str, state)),
                    "action": action,
                    "training_state_visits": visits,
                    "training_action_updates": int(count),
                }
            )
    update_counts = pd.DataFrame(update_rows)

    margin_rows = []
    for _, row in learned_history.iterrows():
        available = [
            action for action in ACTION_ORDER
            if int(row.get(f"available_{action}", 0)) == 1
        ]
        values = np.asarray([float(row[f"q_{action}"]) for action in available])
        if len(values) == 0:
            continue
        ordered = np.sort(values)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float("nan")
        margin_rows.append(
            {
                "seed": row["seed"],
                "scenario": row["scenario"],
                "round": row["round"],
                "q_state_key": row.get("q_state_key", ""),
                "q_state_seen": int(row.get("q_state_seen", 0)),
                "available_actions": len(available),
                "best_q_margin": margin,
                "near_tie": int(np.isfinite(margin) and margin <= 1e-6),
                "planner_action": row["planner_action"],
                "planner_reward": row["planner_reward"],
            }
        )
    margins = pd.DataFrame(margin_rows)

    coverage = pd.DataFrame(
        [
            {
                "evaluation_rounds": len(margins),
                "unique_evaluation_states": margins["q_state_key"].nunique(),
                "unseen_evaluation_round_share": 1.0 - float(margins["q_state_seen"].mean()),
                "near_tie_round_share": float(margins["near_tie"].mean()),
                "mean_best_q_margin": float(margins["best_q_margin"].mean()),
                "median_best_q_margin": float(margins["best_q_margin"].median()),
                "trained_q_states": len(planner.q_table),
            }
        ]
    )
    return update_counts, margins, coverage


def write_report(
    *,
    markdown_dir: Path,
    policy_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    coverage: pd.DataFrame,
    eval_seeds: int,
    model_path: Path,
) -> None:
    overall = (
        policy_summary.groupby(["policy_name", "policy_label"], observed=False)
        ["mean_final_welfare"]
        .mean()
        .sort_values(ascending=False)
    )
    lines = [
        "# Adaptive baseline evaluation",
        "",
        "The learned policy and comparison policies are evaluated on the same simulation seeds. "
        "The main comparison is learned Q versus uniform random feasible institutional choice; "
        "frequency-informed random feasible and shuffled learned sequence are ex-post sequencing diagnostics.",
        "",
        f"Model: `{model_path.relative_to(REPO_ROOT)}`.",
        f"Evaluation seeds per scenario: `{eval_seeds}`.",
        "",
        "## Mean final planner welfare",
        "",
    ]
    for (policy_name, policy_label), value in overall.items():
        lines.append(f"- {policy_label} (`{policy_name}`): {value:.3f}")

    lines.extend(["", "## Learned Q minus comparison policies", ""])
    subset = comparisons[
        (comparisons["scenario"] == "ALL")
        & (comparisons["metric"] == "final_welfare")
    ].sort_values("baseline_policy")
    for _, row in subset.iterrows():
        lines.append(
            f"- vs `{row['baseline_policy']}`: mean difference {row['mean_target_minus_baseline']:.3f}, "
            f"seed-clustered 95% CI [{row['seed_clustered_ci95_low']:.3f}, "
            f"{row['seed_clustered_ci95_high']:.3f}]"
        )

    cov = coverage.iloc[0]
    lines.extend(
        [
            "",
            "## State coverage",
            "",
            f"- Trained Q states: {int(cov['trained_q_states'])}.",
            f"- Unique evaluation states: {int(cov['unique_evaluation_states'])}.",
            f"- Unseen evaluation round share: {cov['unseen_evaluation_round_share']:.4f}.",
            f"- Median best-versus-second feasible Q margin: {cov['median_best_q_margin']:.4f}.",
            "",
            "The pooled comparison does not establish a welfare advantage from the learned "
            "state-to-action mapping over simple feasible multi-institution choice.",
        ]
    )
    (markdown_dir / "adaptive_baseline_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the learned Q policy against paired institutional baselines."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-seeds", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=2000, help="Scenario-level paired bootstrap draws.")
    parser.add_argument("--cluster-bootstrap-draws", type=int, default=10000, help="Seed-clustered bootstrap draws for pooled comparisons.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_root = Path(args.output_root)
    csv_dir, tables_dir, markdown_dir = prepare_output_dirs(output_root)

    planner = OnlineTabularQPlanner.load(model_path)
    learned_policy = LearnedQPolicy(planner)
    learned_summary, learned_history, learned_agents = evaluate_policy(
        policy=learned_policy,
        policy_index=0,
        eval_seeds=args.eval_seeds,
    )
    counts = Counter(learned_history["planner_action"])
    sequences = learned_sequences(learned_history)
    policies: list[PlannerPolicy] = [
        PermanentActionPolicy(),
        UniformRandomFeasiblePolicy(),
        FrequencyMatchedRandomPolicy(counts),
        ShuffledLearnedTimingPolicy(
            action_sequences=sequences,
            fallback_counts=counts,
        ),
    ]

    summaries = [learned_summary]
    histories = [learned_history]
    agents = [learned_agents]
    for index, policy in enumerate(policies, start=1):
        print(f"evaluating {policy.name}")
        summary, history, final_agents = evaluate_policy(
            policy=policy,
            policy_index=index,
            eval_seeds=args.eval_seeds,
        )
        summaries.append(summary)
        histories.append(history)
        agents.append(final_agents)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_history = pd.concat(histories, ignore_index=True)
    all_agents = pd.concat(agents, ignore_index=True)
    all_summary.to_csv(csv_dir / "policy_seed_results.csv", index=False)
    all_history.to_csv(csv_dir / "policy_round_history.csv", index=False)
    all_agents.to_csv(csv_dir / "policy_final_agents.csv", index=False)

    policy_summary = summarize_policy_outcomes(all_summary)
    allocation = action_allocation(all_summary)
    comparisons = paired_comparisons(
        all_summary,
        bootstrap_draws=args.bootstrap_draws,
        cluster_bootstrap_draws=args.cluster_bootstrap_draws,
    )
    update_counts, margins, coverage = q_state_diagnostics(
        planner=planner,
        learned_history=learned_history,
    )
    policy_summary.to_csv(tables_dir / "policy_summary_by_scenario.csv", index=False)
    allocation.to_csv(tables_dir / "action_allocation_by_policy.csv", index=False)
    comparisons.to_csv(tables_dir / "paired_policy_comparisons.csv", index=False)
    update_counts.to_csv(tables_dir / "state_action_update_counts.csv", index=False)
    margins.to_csv(tables_dir / "learned_q_margin_by_round.csv", index=False)
    coverage.to_csv(tables_dir / "state_signal_coverage_summary.csv", index=False)
    (csv_dir / "baseline_metadata.json").write_text(
        json.dumps(
            {
                "model": display_path(model_path),
                "output_root": display_path(output_root),
                "eval_seeds": args.eval_seeds,
                "eval_seed_offset": EVAL_SEED_OFFSET,
                "scenario_bootstrap_draws": args.bootstrap_draws,
                "cluster_bootstrap_draws": args.cluster_bootstrap_draws,
                "policies": sorted(all_summary["policy_name"].unique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        markdown_dir=markdown_dir,
        policy_summary=policy_summary,
        comparisons=comparisons,
        coverage=coverage,
        eval_seeds=args.eval_seeds,
        model_path=model_path,
    )
    print(f"wrote adaptive baseline outputs to {output_root}")


if __name__ == "__main__":
    main()
