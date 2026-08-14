"""Analyze the online institutional Q-learning experiment.

The analysis intentionally contains no front-loaded, back-loaded, periodic, or
other scheduled institutional baselines. It focuses on the learned policy's
welfare outcomes, action allocation, realized governance costs, and capacity
trajectories across behavioral scenarios.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs" / ".internal"
MPLCONFIGDIR = INTERNAL_OUTPUT_ROOT / ".mplconfig"
XDG_CACHE_HOME = INTERNAL_OUTPUT_ROOT / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT = (
    REPO_ROOT
    / "experiments"
    / "outputs"
    / "adaptive_planner"
)

SCENARIO_ORDER = [
    "need_based",
    "cooperative",
    "hoarding",
    "mixed",
    "need_based_to_competitive",
    "cooperative_to_hoarding",
]
SCENARIO_LABELS = {
    "need_based": "Need-based",
    "cooperative": "Cooperative",
    "hoarding": "Hoarding",
    "mixed": "Mixed",
    "need_based_to_competitive": "Need-based → competitive",
    "cooperative_to_hoarding": "Cooperative → hoarding",
}
ACTION_ORDER = [
    "bilateral_3pass",
    "clearinghouse",
    "public_pool",
    "subsidized_catch_up",
    "central_cap2",
    "equity_cap2",
    "central_full",
]
ACTION_LABELS = {
    "bilateral_3pass": "Bilateral 3-pass",
    "clearinghouse": "Clearinghouse",
    "public_pool": "Public pool",
    "subsidized_catch_up": "Subsidized catch-up",
    "central_cap2": "Central cap 2",
    "equity_cap2": "Equity cap 2",
    "central_full": "Full central",
}


def prepare_dirs(root: Path) -> tuple[Path, Path, Path]:
    tables = root / "tables"
    plots = root / "plots" / "main"
    markdown = root / "markdown"
    for path in [tables, plots, markdown]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return tables, plots, markdown


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_dir = root / "csv"
    paths = {
        "training": csv_dir / "training_episodes.csv",
        "summary": csv_dir / "evaluation_summary_by_seed.csv",
        "history": csv_dir / "evaluation_round_history.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing online-planner output files:\n- " + "\n- ".join(missing)
        )
    training = pd.read_csv(paths["training"])
    summary = pd.read_csv(paths["summary"])
    history = pd.read_csv(paths["history"])

    required_summary = {
        "seed",
        "scenario",
        "final_total_score",
        "final_min_score",
        "final_score_gap",
        "final_welfare",
        "cumulative_capacity_cost",
    }
    required_history = {
        "seed",
        "scenario",
        "round",
        "planner_action",
        "coordination_capacity_before",
        "coordination_capacity_after",
        "capacity_realized_cost",
        "institution_workload_units",
    }
    if required_summary - set(summary.columns):
        raise ValueError(
            f"Evaluation summary missing columns: {sorted(required_summary - set(summary.columns))}"
        )
    if required_history - set(history.columns):
        raise ValueError(
            f"Evaluation history missing columns: {sorted(required_history - set(history.columns))}"
        )
    if summary.duplicated(["seed", "scenario"]).any():
        raise ValueError("Duplicate evaluation seed-scenario keys found.")
    return training, summary, history


def build_tables(
    *,
    root: Path,
    summary: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = root / "tables"

    scenario_means = (
        summary.groupby(["scenario", "scenario_label"], observed=False)
        .agg(
            n=("seed", "nunique"),
            mean_final_total_score=("final_total_score", "mean"),
            mean_final_min_score=("final_min_score", "mean"),
            mean_final_score_gap=("final_score_gap", "mean"),
            mean_final_welfare=("final_welfare", "mean"),
            mean_capacity_cost=("cumulative_capacity_cost", "mean"),
            mean_final_capacity=("final_coordination_capacity", "mean"),
        )
        .reset_index()
    )
    scenario_means.to_csv(tables / "scenario_outcomes.csv", index=False)

    allocation = (
        summary.groupby(["scenario", "scenario_label"], observed=False)[
            [f"rounds_{action}" for action in ACTION_ORDER]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={f"rounds_{action}": action for action in ACTION_ORDER}
        )
    )
    allocation.to_csv(tables / "action_allocation_by_scenario.csv", index=False)

    # Score and minimum-score gains are available from pre-observation columns.
    institution_costs = (
        history.assign(
            score_gain=(
                history["total_score"] - history["observation_total_score"]
            ),
            min_score_gain=(
                history["min_score"] - history["observation_min_score"]
            ),
        )
        .groupby(
            ["planner_action", "planner_action_label"],
            observed=False,
        )
        .agg(
            selected_rounds=("round", "count"),
            mean_realized_capacity_cost=("capacity_realized_cost", "mean"),
            mean_workload_units=("institution_workload_units", "mean"),
            mean_round_reward=("planner_reward", "mean"),
            mean_score_gain=("score_gain", "mean"),
            mean_min_score_gain=("min_score_gain", "mean"),
        )
        .reset_index()
    )
    institution_costs.to_csv(tables / "institution_costs_and_returns.csv", index=False)

    capacity = (
        history.groupby(
            ["scenario", "scenario_label", "round"], observed=False
        )
        .agg(
            mean_capacity_before=("coordination_capacity_before", "mean"),
            mean_capacity_after=("coordination_capacity_after", "mean"),
            mean_capacity_cost=("capacity_realized_cost", "mean"),
        )
        .reset_index()
    )
    capacity.to_csv(tables / "capacity_trajectory.csv", index=False)
    return scenario_means, allocation, institution_costs, capacity


def plot_training_progress(training: pd.DataFrame, path: Path) -> None:
    ordered = training.sort_values("episode").copy()
    window = max(20, min(300, len(ordered) // 20 if len(ordered) else 20))
    ordered["rolling_welfare"] = ordered["final_welfare"].rolling(
        window, min_periods=max(2, window // 5)
    ).mean()

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.plot(ordered["episode"], ordered["rolling_welfare"])
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rolling mean final welfare")
    ax.set_title("Online learning progress", pad=12)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_action_allocation(allocation: pd.DataFrame, path: Path) -> None:
    data = allocation.copy()
    data["scenario"] = pd.Categorical(
        data["scenario"], SCENARIO_ORDER, ordered=True
    )
    data = data.sort_values("scenario")
    x = np.arange(len(data))
    bottom = np.zeros(len(data))

    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    for action in ACTION_ORDER:
        values = data[action].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=ACTION_LABELS[action])
        bottom += values
    ax.set_xticks(x, data["scenario_label"], rotation=18, ha="right")
    ax.set_ylabel("Mean rounds using institution")
    ax.set_ylim(0, 20)
    ax.set_title("Learned institutional mix across behavioral scenarios", pad=14)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.31, top=0.88)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_trajectory(capacity: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    for scenario in SCENARIO_ORDER:
        line = capacity[capacity["scenario"] == scenario].sort_values("round")
        if line.empty:
            continue
        ax.plot(
            line["round"],
            line["mean_capacity_before"],
            marker="o",
            markersize=3,
            label=SCENARIO_LABELS[scenario],
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean coordination capacity before choice")
    ax.set_title("Real institutional capacity is depleted and regenerated", pad=12)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_outcomes(outcomes: pd.DataFrame, path: Path) -> None:
    data = outcomes.copy()
    data["scenario"] = pd.Categorical(
        data["scenario"], SCENARIO_ORDER, ordered=True
    )
    data = data.sort_values("scenario")
    x = np.arange(len(data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    ax.bar(
        x - width / 2,
        data["mean_final_total_score"],
        width,
        label="Total score",
    )
    ax.bar(
        x + width / 2,
        data["mean_final_min_score"],
        width,
        label="Weakest-agent score",
    )
    ax.set_xticks(x, data["scenario_label"], rotation=18, ha="right")
    ax.set_ylabel("Mean final score")
    ax.set_title("Development and weakest-agent outcomes of the learned planner", pad=14)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.88)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transition_policy(history: pd.DataFrame, scenario: str, path: Path) -> None:
    subset = history[history["scenario"] == scenario]
    shares = (
        subset.groupby(["round", "planner_action"], observed=False)
        .size()
        .unstack("planner_action", fill_value=0)
    )
    shares = shares.div(shares.sum(axis=1), axis=0)
    for action in ACTION_ORDER:
        if action not in shares.columns:
            shares[action] = 0.0
    shares = shares[ACTION_ORDER]

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    bottom = np.zeros(len(shares))
    x = shares.index.to_numpy()
    for action in ACTION_ORDER:
        values = shares[action].to_numpy()
        ax.bar(x, values, bottom=bottom, label=ACTION_LABELS[action], width=0.85)
        bottom += values
    ax.axvline(9.5, linestyle="--", linewidth=1.2)
    ax.text(9.7, 1.02, "behavior changes", va="bottom", fontsize=9)
    ax.set_xlabel("Round")
    ax.set_ylabel("Share of held-out seeds")
    ax.set_ylim(0, 1)
    ax.set_title(SCENARIO_LABELS[scenario], loc="left", pad=16)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.29, top=0.85)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_memo(
    *,
    root: Path,
    outcomes: pd.DataFrame,
    allocation: pd.DataFrame,
    institution_costs: pd.DataFrame,
) -> None:
    markdown = root / "markdown"
    outcome_index = outcomes.set_index("scenario")
    allocation_index = allocation.set_index("scenario")
    lines = [
        "# Online institutional planner results",
        "",
        "The planner was trained online through direct interaction with simulated games. There are no scheduled institutional baselines in this experiment. Governance capacity is part of the environment state: coordinated institutions deplete it according to base, switching, and realized workload costs, while unused capacity regenerates between rounds.",
        "",
        "## Held-out outcomes",
        "",
    ]
    for scenario in SCENARIO_ORDER:
        if scenario not in outcome_index.index:
            continue
        row = outcome_index.loc[scenario]
        allocation_row = allocation_index.loc[scenario]
        most_used = max(ACTION_ORDER, key=lambda action: allocation_row.get(action, 0.0))
        lines.append(
            f"- {SCENARIO_LABELS[scenario]}: total score {row['mean_final_total_score']:.2f}; "
            f"weakest-agent score {row['mean_final_min_score']:.2f}; "
            f"welfare {row['mean_final_welfare']:.2f}; "
            f"mean capacity cost {row['mean_capacity_cost']:.2f}; "
            f"most-used institution {ACTION_LABELS[most_used]}."
        )

    lines.extend([
        "",
        "## Institutional use and cost",
        "",
    ])
    for _, row in institution_costs.sort_values("selected_rounds", ascending=False).iterrows():
        lines.append(
            f"- {row['planner_action_label']}: selected {int(row['selected_rounds'])} held-out rounds; "
            f"mean realized capacity cost {row['mean_realized_capacity_cost']:.2f}; "
            f"mean workload {row['mean_workload_units']:.2f} resource-handling units; "
            f"mean welfare reward {row['mean_round_reward']:.3f}."
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The main result is whether the learned institutional mix changes coherently with behavioral composition, bottlenecks, inequality, and available governance capacity. The fixed-institution benchmark remains the external reference for how the individual institutions perform when selected continuously; it is not repeated as a scheduled switching experiment here.",
    ])
    (markdown / "online_planner_results_memo.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def regenerate_online_planner_outputs(
    *,
    output_root: str | Path = DEFAULT_ROOT,
) -> None:
    root = Path(output_root)
    prepare_dirs(root)
    training, summary, history = load_data(root)
    outcomes, allocation, institution_costs, capacity = build_tables(
        root=root,
        summary=summary,
        history=history,
    )
    plots = root / "plots" / "main"
    plot_training_progress(training, plots / "01_training_progress.png")
    plot_action_allocation(allocation, plots / "02_action_allocation.png")
    plot_capacity_trajectory(capacity, plots / "03_capacity_trajectory.png")
    plot_scenario_outcomes(outcomes, plots / "04_scenario_outcomes.png")
    plot_transition_policy(
        history,
        "need_based_to_competitive",
        plots / "05_policy_need_based_to_competitive.png",
    )
    plot_transition_policy(
        history,
        "cooperative_to_hoarding",
        plots / "06_policy_cooperative_to_hoarding.png",
    )
    write_memo(
        root=root,
        outcomes=outcomes,
        allocation=allocation,
        institution_costs=institution_costs,
    )

    print("Online-planner analysis complete")
    print(f"Evaluation rows: {len(summary):,}")
    print(
        "Matched seeds per scenario: "
        f"{summary.groupby('scenario')['seed'].nunique().min()}"
    )
    print("\nMean held-out outcomes:")
    for _, row in outcomes.iterrows():
        print(
            f"- {row['scenario_label']}: total={row['mean_final_total_score']:.2f}, "
            f"min={row['mean_final_min_score']:.2f}, "
            f"welfare={row['mean_final_welfare']:.2f}, "
            f"capacity cost={row['mean_capacity_cost']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate analysis for an existing online-planner output folder."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_ROOT),
        help=(
            "Planner output directory containing csv/, tables/, plots/, and markdown/. "
            "Defaults to the adaptive-planner output used in the paper."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    regenerate_online_planner_outputs(output_root=args.output_root)
