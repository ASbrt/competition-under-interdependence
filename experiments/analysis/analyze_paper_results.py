"""Generate the compact result tables used by the paper.

This script does not rerun simulations. It reads the fixed-institution,
composition-sweep, adaptive-planner, and paired-baseline outputs and applies the
uncertainty calculations described in the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adaptive.online_q_planner import ACTION_ORDER  # noqa: E402

OUT = ROOT / "experiments" / "outputs" / "paper_results"
TABLES = OUT / "tables"

POLICY_RESULTS = OUT.parent / "adaptive_baselines" / "csv" / "policy_seed_results.csv"
BASELINE_COMPARISONS = OUT.parent / "adaptive_baselines" / "tables" / "paired_policy_comparisons.csv"
ADAPTIVE_HISTORY = OUT.parent / "adaptive_planner" / "csv" / "evaluation_round_history.csv"
COMPOSITION = OUT.parent / "composition_sweep" / "csv" / "summary_by_seed.csv"
FIXED_SUMMARY = OUT.parent / "fixed_institution" / "csv" / "summary_by_seed.csv"
BRANCH_SUMMARY = OUT.parent / "branch_diagnostic" / "branch_aggregate_summary.csv"

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260811

def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy so downstream analysis never mutates stored source data."""
    return df.copy()


def seed_cluster_ci(diff_rows: pd.DataFrame, *, rng: np.random.Generator) -> tuple[float, float]:
    cluster_stats = [
        (float(rows["diff"].sum()), len(rows))
        for _, rows in diff_rows.groupby("seed", observed=False)
    ]
    sums = np.asarray([item[0] for item in cluster_stats], dtype=np.float64)
    counts = np.asarray([item[1] for item in cluster_stats], dtype=np.float64)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(sums), size=len(sums))
        draws[draw] = float(sums[sampled].sum() / counts[sampled].sum())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_policy_diff(
    summary: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    metric: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    target_rows = summary[summary["policy_name"] == target][["seed", "scenario", metric]]
    baseline_rows = summary[summary["policy_name"] == baseline][["seed", "scenario", metric]]
    merged = target_rows.merge(
        baseline_rows,
        on=["seed", "scenario"],
        suffixes=("_target", "_baseline"),
    )
    merged["diff"] = merged[f"{metric}_target"] - merged[f"{metric}_baseline"]
    low, high = seed_cluster_ci(merged[["seed", "scenario", "diff"]], rng=rng)
    return {
        "target_policy": target,
        "baseline_policy": baseline,
        "metric": metric,
        "mean_difference": float(merged["diff"].mean()),
        "seed_clustered_ci95_low": low,
        "seed_clustered_ci95_high": high,
        "seed_clusters": int(merged["seed"].nunique()),
        "observations": int(len(merged)),
    }


def policy_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    baselines = [
        "permanent_bilateral_3pass",
        "random_feasible",
        "frequency_informed_random",
        "shuffled_learned_sequence",
    ]
    rows = [
        paired_policy_diff(
            summary,
            target="learned_q",
            baseline=baseline,
            metric="final_welfare",
            rng=rng,
        )
        for baseline in baselines
    ]
    rows.extend(
        [
            paired_policy_diff(
                summary,
                target="frequency_informed_random",
                baseline=baseline,
                metric="final_welfare",
                rng=rng,
            )
            for baseline in ["permanent_bilateral_3pass", "random_feasible"]
        ]
    )
    return pd.DataFrame(rows)


def scenario_q_vs_random() -> pd.DataFrame:
    if not BASELINE_COMPARISONS.exists():
        return pd.DataFrame()
    df = normalize_names(pd.read_csv(BASELINE_COMPARISONS))
    mean_column = (
        "mean_target_minus_baseline"
        if "mean_target_minus_baseline" in df.columns
        else "mean_learned_minus_baseline"
    )
    sub = df[
        (df["baseline_policy"] == "random_feasible")
        & (df["metric"] == "final_welfare")
        & (df["scenario"] != "ALL")
    ][["scenario", mean_column, "ci95_low", "ci95_high"]].copy()
    return sub.rename(columns={mean_column: "mean_difference"})


def realized_action_shares(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [
        "learned_q",
        "frequency_informed_random",
        "shuffled_learned_sequence",
        "random_feasible",
    ]
    action_cols = [f"rounds_{action}" for action in ACTION_ORDER]
    pooled_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    for policy, rows in summary[summary["policy_name"].isin(policies)].groupby("policy_name", observed=False):
        counts = rows[action_cols].sum()
        total = float(counts.sum())
        pooled_rows.extend(
            {
                "policy_name": policy,
                "action": action,
                "rounds": int(counts[f"rounds_{action}"]),
                "share": float(counts[f"rounds_{action}"] / total),
            }
            for action in ACTION_ORDER
        )
        for scenario, scenario_data in rows.groupby("scenario", observed=False):
            scenario_counts = scenario_data[action_cols].sum()
            scenario_total = float(scenario_counts.sum())
            scenario_rows.extend(
                {
                    "policy_name": policy,
                    "scenario": scenario,
                    "action": action,
                    "rounds": int(scenario_counts[f"rounds_{action}"]),
                    "share": float(scenario_counts[f"rounds_{action}"] / scenario_total),
                }
                for action in ACTION_ORDER
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(scenario_rows)


def welfare_scale(summary: pd.DataFrame) -> pd.DataFrame:
    policies = [
        "learned_q",
        "permanent_bilateral_3pass",
        "random_feasible",
        "frequency_informed_random",
        "shuffled_learned_sequence",
    ]
    rows = []
    for policy, data in summary[summary["policy_name"].isin(policies)].groupby("policy_name", observed=False):
        values = data["final_welfare"].to_numpy(float)
        rows.append(
            {
                "policy_name": policy,
                "n": len(values),
                "mean_final_welfare": float(np.mean(values)),
                "sd_final_welfare": float(np.std(values, ddof=1)),
                "p25_final_welfare": float(np.quantile(values, 0.25)),
                "median_final_welfare": float(np.median(values)),
                "p75_final_welfare": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def unseen_state_summary(history: pd.DataFrame) -> pd.DataFrame:
    unseen = history[history["q_state_seen"] == 0]
    actions = unseen["planner_action"].value_counts().rename_axis("action").reset_index(name="n")
    return pd.DataFrame(
        [
            {
                "evaluation_decisions": int(len(history)),
                "unseen_decisions": int(len(unseen)),
                "unseen_share": float(len(unseen) / len(history)),
                "unseen_actions": "; ".join(f"{row.action}:{row.n}" for row in actions.itertuples()),
            }
        ]
    )


def composition_sensitivity() -> pd.DataFrame:
    df = pd.read_csv(COMPOSITION)
    conditions = [
        "Bilateral 3-pass",
        "Public pool",
        "Central capped (2)",
        "Central clearing",
        "No trade",
    ]
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for (sweep, condition), data in df[df["condition_label"].isin(conditions)].groupby(
        ["sweep_label", "condition_label"], observed=False
    ):
        endpoints = data[data["restrictive_count"].isin([0, 5])]
        pivot = endpoints.pivot(index="seed", columns="restrictive_count", values="final_total_score").dropna()
        diffs = pivot[5] - pivot[0]
        draws = np.asarray(
            [float(rng.choice(diffs, size=len(diffs), replace=True).mean()) for _ in range(BOOTSTRAP_DRAWS)]
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "sweep_label": sweep,
                "condition_label": condition,
                "n_matched_seeds": int(len(diffs)),
                "mean_change_5_minus_0": float(diffs.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def central_equity() -> pd.DataFrame:
    df = pd.read_csv(FIXED_SUMMARY)
    df = df[df["build_mode"] == "development_oriented"]
    metrics = {
        "final_total_score": "final total score",
        "final_score_gap": "final max-min score range",
    }
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    for metric, definition in metrics.items():
        pivot = (
            df[df["condition"].isin(["central_clearing", "equity_weighted_central"])]
            .pivot(index=["seed", "population"], columns="condition", values=metric)
            .dropna()
            .reset_index()
        )
        pivot["diff"] = pivot["central_clearing"] - pivot["equity_weighted_central"]
        low, high = seed_cluster_ci(pivot[["seed", "population", "diff"]], rng=rng)
        rows.append(
            {
                "metric": metric,
                "metric_definition": definition,
                "comparison": "central_clearing_minus_equity_central",
                "mean_difference": float(pivot["diff"].mean()),
                "seed_clustered_ci95_low": low,
                "seed_clustered_ci95_high": high,
                "seed_clusters": int(pivot["seed"].nunique()),
                "observations": int(len(pivot)),
            }
        )
    return pd.DataFrame(rows)


def build_mode_comparison() -> pd.DataFrame:
    df = pd.read_csv(FIXED_SUMMARY)
    pivot = (
        df.pivot(
            index=["seed", "population", "condition"],
            columns="build_mode",
            values="final_total_score",
        )
        .dropna()
        .reset_index()
    )
    pivot["diff"] = pivot["crown_aware"] - pivot["development_oriented"]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 3)
    low, high = seed_cluster_ci(pivot[["seed", "population", "condition", "diff"]], rng=rng)
    return pd.DataFrame(
        [
            {
                "comparison": "crown_aware_minus_development_oriented",
                "mean_difference": float(pivot["diff"].mean()),
                "seed_clustered_ci95_low": low,
                "seed_clustered_ci95_high": high,
                "seed_clusters": int(pivot["seed"].nunique()),
                "observations": int(len(pivot)),
            }
        ]
    )


def seed_design() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "training_seed_range": "100000-111999",
                "training_episodes": 12000,
                "evaluation_seed_range": "500000-500099",
                "evaluation_seed_clusters": 100,
                "evaluation_disjoint_from_training": True,
                "evaluation_set_reused_for_additional_learner_variants": True,
            }
        ]
    )


def write_summary(tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Paper result tables",
        "",
        "These tables are generated from the stored experiment outputs using the uncertainty procedures described in the paper.",
        "",
        "## Adaptive policy comparisons",
        "",
        tables["policy_comparisons"].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Behavioral-composition endpoint comparisons",
        "",
        tables["composition_sensitivity"].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Central versus equity-oriented central coordination",
        "",
        tables["central_equity"].to_markdown(index=False, floatfmt=".3f"),
    ]
    if BRANCH_SUMMARY.exists():
        branch = normalize_names(pd.read_csv(BRANCH_SUMMARY))
        lines.extend(["", "## Counterfactual branch diagnostic", "", branch.to_markdown(index=False, floatfmt=".3f")])
    (OUT / "paper_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary = normalize_names(pd.read_csv(POLICY_RESULTS))
    history = normalize_names(pd.read_csv(ADAPTIVE_HISTORY))

    tables: dict[str, pd.DataFrame] = {
        "policy_comparisons": policy_comparisons(summary),
        "scenario_q_vs_random": scenario_q_vs_random(),
        "welfare_scale": welfare_scale(summary),
        "unseen_states": unseen_state_summary(history),
        "composition_sensitivity": composition_sensitivity(),
        "central_equity": central_equity(),
        "build_mode_comparison": build_mode_comparison(),
        "seed_design": seed_design(),
    }
    pooled, scenario = realized_action_shares(summary)
    tables["action_shares_pooled"] = pooled
    tables["action_shares_by_scenario"] = scenario

    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)

    metadata = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_outputs": [
            "experiments/outputs/fixed_institution",
            "experiments/outputs/composition_sweep",
            "experiments/outputs/adaptive_planner",
            "experiments/outputs/adaptive_baselines",
            "experiments/outputs/branch_diagnostic",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_summary(tables)
    print(f"wrote paper result tables to {OUT}")


if __name__ == "__main__":
    main()
