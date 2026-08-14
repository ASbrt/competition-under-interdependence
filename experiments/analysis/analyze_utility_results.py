"""Post-process the fixed-institution benchmark.

This script reads the seed-level and final-agent outputs, validates the matched
factorial design, and writes compact tables and diagnostic plots.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import sys
from statistics import NormalDist
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "outputs" / "fixed_institution"
SOURCE_CSV_DIR = EXPERIMENT_ROOT / "csv"
TABLES_DIR = EXPERIMENT_ROOT / "tables"
MARKDOWN_DIR = EXPERIMENT_ROOT / "markdown"
MAIN_PLOTS_DIR = EXPERIMENT_ROOT / "plots" / "main"
SUPPORTING_PLOTS_DIR = EXPERIMENT_ROOT / "plots" / "supporting"
INTERNAL_OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs" / ".internal"
MPLCONFIGDIR = INTERNAL_OUTPUT_ROOT / ".mplconfig"
XDG_CACHE_HOME = INTERNAL_OUTPUT_ROOT / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except ModuleNotFoundError:
    student_t = None

from experiments.analysis.utility_plotting import (
    INSTITUTION_LABELS,
    INSTITUTION_ORDER,
    POPULATION_LABELS,
    POPULATION_ORDER,
    apply_report_style,
    plot_crown_effect_supporting,
    plot_development_score_heatmap,
    plot_efficiency_weakest_agent_frontier,
    plot_paired_public_pool_vs_central,
    plot_robustness_frontier,
    plot_selected_interaction,
)

BUILD_MODES = ["development_oriented", "crown_aware"]

INSTITUTION_FAMILIES = {
    "no_trade": ("baseline", "Baseline"),
    "bilateral_trade": ("decentralized_bargaining", "Decentralized bargaining"),
    "bilateral_trade_3pass": ("decentralized_bargaining", "Decentralized bargaining"),
    "catch_up_bilateral_trade": ("decentralized_bargaining", "Decentralized bargaining"),
    "bottleneck_priority_bilateral_trade": ("decentralized_bargaining", "Decentralized bargaining"),
    "clearinghouse_bargaining": ("coordinated_bargaining", "Coordinated bargaining"),
    "subsidized_catch_up": ("redistributive_support", "Redistributive support"),
    "public_pool": ("contribution_pool", "Contribution-dependent pool"),
    "central_clearing": ("central_matching", "Central matching"),
    "equity_weighted_central": ("central_matching", "Central matching"),
    "central_clearing_capped": ("central_matching", "Central matching"),
}

SUMMARY_METRICS = [
    "final_total_score",
    "final_score_gap",
    "cumulative_trades_executed",
    "final_idle_infrastructure",
    "final_total_infrastructure",
    "final_total_production_sites",
    "final_total_advanced_sites",
    "final_total_innovation",
]


def prepare_output_dirs() -> None:
    """Recreate derived outputs while leaving source CSVs untouched."""
    for path in [TABLES_DIR, MARKDOWN_DIR, MAIN_PLOTS_DIR, SUPPORTING_PLOTS_DIR]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return float("nan")
    if student_t is not None:
        return float(student_t.ppf(0.975, degrees_of_freedom))
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    clean = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    n = int(clean.size)
    mean = float(clean.mean()) if n else float("nan")
    sd = float(clean.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    margin = _t_critical_975(n - 1) * se if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": se,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "median": float(clean.median()) if n else float("nan"),
    }


def load_and_validate_sources() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load source tables and validate a complete matched-seed factorial."""
    paths = {
        "summary": SOURCE_CSV_DIR / "summary_by_seed.csv",
        "agents": SOURCE_CSV_DIR / "final_agents.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing fixed-institution source CSVs:\n- " + "\n- ".join(missing)
        )

    summary = pd.read_csv(paths["summary"])
    agents = pd.read_csv(paths["agents"])

    required_summary = {
        "build_mode",
        "population",
        "condition",
        "seed",
        *SUMMARY_METRICS,
    }
    required_agents = {
        "build_mode",
        "population",
        "condition",
        "seed",
        "agent_id",
        "score",
    }
    missing_summary = required_summary - set(summary.columns)
    missing_agents = required_agents - set(agents.columns)
    if missing_summary:
        raise ValueError(f"summary_by_seed.csv is missing columns: {sorted(missing_summary)}")
    if missing_agents:
        raise ValueError(f"final_agents.csv is missing columns: {sorted(missing_agents)}")

    if set(summary["build_mode"]) != set(BUILD_MODES):
        raise ValueError(f"Unexpected build modes: {sorted(summary['build_mode'].unique())}")
    if set(summary["population"]) != set(POPULATION_ORDER):
        raise ValueError(f"Unexpected populations: {sorted(summary['population'].unique())}")
    if set(summary["condition"]) != set(INSTITUTION_ORDER):
        raise ValueError(f"Unexpected institutions: {sorted(summary['condition'].unique())}")

    key_columns = ["build_mode", "population", "condition", "seed"]
    if summary.duplicated(key_columns).any():
        raise ValueError("summary_by_seed.csv contains duplicate factorial keys.")

    n_seeds = int(summary["seed"].nunique())
    expected_summary_rows = len(BUILD_MODES) * len(POPULATION_ORDER) * len(INSTITUTION_ORDER) * n_seeds
    if len(summary) != expected_summary_rows:
        raise ValueError(
            f"Expected {expected_summary_rows:,} summary rows for {n_seeds} seeds, "
            f"found {len(summary):,}."
        )

    seeds_per_cell = summary.groupby(["build_mode", "population", "condition"])["seed"].nunique()
    if not seeds_per_cell.eq(n_seeds).all():
        raise ValueError("At least one factorial cell has incomplete seed coverage.")

    expected_agent_rows = expected_summary_rows * int(agents["agent_id"].nunique())
    if len(agents) != expected_agent_rows:
        raise ValueError(
            f"Expected {expected_agent_rows:,} final-agent rows, found {len(agents):,}."
        )

    return summary, agents, n_seeds


def attach_family_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["institution_family"] = result["condition"].map(
        lambda value: INSTITUTION_FAMILIES[value][0]
    )
    result["institution_family_label"] = result["condition"].map(
        lambda value: INSTITUTION_FAMILIES[value][1]
    )
    return result


def build_mean_cells(summary: pd.DataFrame) -> pd.DataFrame:
    """Create development-oriented cell means and seed-level uncertainty."""
    development = summary[summary["build_mode"] == "development_oriented"].copy()
    rows: list[dict[str, object]] = []
    for (population, condition), group in development.groupby(
        ["population", "condition"], observed=False
    ):
        row: dict[str, object] = {
            "population": population,
            "population_label": POPULATION_LABELS[population],
            "condition": condition,
            "condition_label": INSTITUTION_LABELS[condition],
        }
        for metric in SUMMARY_METRICS:
            stats = summarize_values(group[metric])
            row[f"mean_{metric}"] = stats["mean"]
            row[f"sd_{metric}"] = stats["standard_deviation"]
            row[f"ci95_low_{metric}"] = stats["ci95_low"]
            row[f"ci95_high_{metric}"] = stats["ci95_high"]
        rows.append(row)
    return attach_family_columns(pd.DataFrame(rows))


def build_agent_seed_outcomes(agents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive weakest-agent, bottom-two, and dispersion outcomes by game."""
    development = agents[agents["build_mode"] == "development_oriented"].copy()

    def bottom_two_mean(values: pd.Series) -> float:
        ordered = values.sort_values().to_numpy()
        return float(np.mean(ordered[: min(2, len(ordered))]))

    def score_gini(values: pd.Series) -> float:
        arr = np.sort(values.to_numpy(dtype=float))
        if len(arr) == 0 or np.isclose(arr.sum(), 0):
            return 0.0
        index = np.arange(1, len(arr) + 1)
        return float((2 * np.sum(index * arr) / (len(arr) * arr.sum())) - (len(arr) + 1) / len(arr))

    game_level = (
        development.groupby(["population", "condition", "seed"], observed=False)
        .agg(
            min_agent_score=("score", "min"),
            mean_agent_score=("score", "mean"),
            score_sd=("score", lambda values: float(values.std(ddof=0))),
            bottom_two_mean_score=("score", bottom_two_mean),
            score_gini=("score", score_gini),
        )
        .reset_index()
    )

    cell_level = (
        game_level.groupby(["population", "condition"], observed=False)
        .agg(
            mean_min_agent_score=("min_agent_score", "mean"),
            mean_bottom_two_score=("bottom_two_mean_score", "mean"),
            mean_score_sd=("score_sd", "mean"),
            mean_score_gini=("score_gini", "mean"),
        )
        .reset_index()
    )
    cell_level["population_label"] = cell_level["population"].map(POPULATION_LABELS)
    cell_level["condition_label"] = cell_level["condition"].map(INSTITUTION_LABELS)
    cell_level = attach_family_columns(cell_level)
    return game_level, cell_level


def build_robustness_table(mean_cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize institution performance across utility populations."""
    result = (
        mean_cells.groupby(
            ["condition", "condition_label", "institution_family", "institution_family_label"],
            observed=False,
        )
        .agg(
            mean_final_total_score=("mean_final_total_score", "mean"),
            worst_population_score=("mean_final_total_score", "min"),
            best_population_score=("mean_final_total_score", "max"),
            standard_deviation_across_populations=("mean_final_total_score", "std"),
        )
        .reset_index()
    )
    result["population_score_range"] = (
        result["best_population_score"] - result["worst_population_score"]
    )
    return result


def build_institution_welfare_table(
    mean_cells: pd.DataFrame,
    agent_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Average aggregate and weakest-agent outcomes across populations."""
    merged = mean_cells.merge(
        agent_cells[
            [
                "population",
                "condition",
                "mean_min_agent_score",
                "mean_bottom_two_score",
                "mean_score_gini",
            ]
        ],
        on=["population", "condition"],
        validate="one_to_one",
    )
    return (
        merged.groupby(
            ["condition", "condition_label", "institution_family", "institution_family_label"],
            observed=False,
        )
        .agg(
            mean_final_total_score=("mean_final_total_score", "mean"),
            mean_min_agent_score=("mean_min_agent_score", "mean"),
            mean_bottom_two_score=("mean_bottom_two_score", "mean"),
            mean_score_gini=("mean_score_gini", "mean"),
        )
        .reset_index()
    )


def build_paired_institution_comparison(
    summary: pd.DataFrame,
    first: str,
    second: str,
) -> pd.DataFrame:
    """Calculate population-specific matched-seed differences first minus second."""
    development = summary[summary["build_mode"] == "development_oriented"]
    selected = development[development["condition"].isin([first, second])]
    pivot = selected.pivot(
        index=["population", "seed"],
        columns="condition",
        values="final_total_score",
    ).reset_index()
    if first not in pivot or second not in pivot:
        raise ValueError(f"Could not pair {first} and {second}.")
    pivot["difference"] = pivot[first] - pivot[second]

    rows: list[dict[str, object]] = []
    for population, group in pivot.groupby("population", observed=False):
        stats = summarize_values(group["difference"])
        rows.append(
            {
                "population": population,
                "population_label": POPULATION_LABELS[population],
                "first_condition": first,
                "first_condition_label": INSTITUTION_LABELS[first],
                "second_condition": second,
                "second_condition_label": INSTITUTION_LABELS[second],
                "difference_definition": f"{first}_minus_{second}",
                "n_paired_seeds": stats["n"],
                "mean_difference": stats["mean"],
                "standard_deviation": stats["standard_deviation"],
                "standard_error": stats["standard_error"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "share_first_higher": float((group["difference"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_crown_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Pair crown-aware and development-oriented scores by seed and cell."""
    keys = ["population", "condition", "seed"]
    development = summary[summary["build_mode"] == "development_oriented"][
        keys + ["final_total_score"]
    ]
    crown = summary[summary["build_mode"] == "crown_aware"][
        keys + ["final_total_score"]
    ]
    paired = crown.merge(
        development,
        on=keys,
        validate="one_to_one",
        suffixes=("_crown", "_development"),
    )
    paired["difference"] = (
        paired["final_total_score_crown"] - paired["final_total_score_development"]
    )

    rows: list[dict[str, object]] = []
    for condition, group in paired.groupby("condition", observed=False):
        stats = summarize_values(group["difference"])
        rows.append(
            {
                "condition": condition,
                "condition_label": INSTITUTION_LABELS[condition],
                "n_paired_observations": stats["n"],
                "mean_difference": stats["mean"],
                "standard_deviation": stats["standard_deviation"],
                "standard_error": stats["standard_error"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
            }
        )
    return pd.DataFrame(rows)


def build_rankings(mean_cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, group in mean_cells.groupby("population", observed=False):
        ordered = group.sort_values(
            ["mean_final_total_score", "condition_label"], ascending=[False, True]
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "population": population,
                    "population_label": POPULATION_LABELS[population],
                    "rank": rank,
                    "condition": row["condition"],
                    "condition_label": row["condition_label"],
                    "mean_final_total_score": row["mean_final_total_score"],
                }
            )
    return pd.DataFrame(rows)


def write_tables(
    mean_cells: pd.DataFrame,
    agent_games: pd.DataFrame,
    agent_cells: pd.DataFrame,
    robustness: pd.DataFrame,
    institution_welfare: pd.DataFrame,
    paired_pool_central: pd.DataFrame,
    crown_effects: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    mean_cells.to_csv(TABLES_DIR / "development_cell_means.csv", index=False)
    agent_games.to_csv(TABLES_DIR / "development_agent_outcomes_by_seed.csv", index=False)
    agent_cells.to_csv(TABLES_DIR / "development_agent_outcomes_by_cell.csv", index=False)
    robustness.to_csv(TABLES_DIR / "institution_robustness_across_populations.csv", index=False)
    institution_welfare.to_csv(TABLES_DIR / "institution_efficiency_and_weakest_agent.csv", index=False)
    paired_pool_central.to_csv(TABLES_DIR / "paired_public_pool_minus_central_clearing.csv", index=False)
    crown_effects.to_csv(TABLES_DIR / "crown_score_effect_by_institution.csv", index=False)
    rankings.to_csv(TABLES_DIR / "development_institution_rankings.csv", index=False)


def write_memo(
    n_seeds: int,
    summary: pd.DataFrame,
    mean_cells: pd.DataFrame,
    robustness: pd.DataFrame,
    paired_pool_central: pd.DataFrame,
    crown_effects: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    top = rankings[rankings["rank"] == 1].copy()
    top_lines = "\n".join(
        f"- **{row.population_label}:** {row.condition_label} "
        f"({row.mean_final_total_score:.2f})"
        for row in top.itertuples()
    )

    most_robust = robustness.sort_values(
        ["worst_population_score", "mean_final_total_score"], ascending=False
    ).iloc[0]
    largest_range = robustness.sort_values("population_score_range", ascending=False).iloc[0]
    coop_pair = paired_pool_central[
        paired_pool_central["population"] == "cooperative"
    ].iloc[0]
    crown_average = float(crown_effects["mean_difference"].mean())

    memo = f"""# Fixed-institution benchmark summary

## Data

- {len(summary):,} completed games;
- {n_seeds} matched seeds per build-mode × population × institution cell;
- seven behavioral populations;
- eleven exchange institutions;
- development-oriented building is the main slice used in the paper.

## Main descriptive result

Central clearing has the highest mean final total score in every behavioral population in the development-oriented slice. The strongest worst-population mean is **{most_robust['condition_label']}** at {most_robust['worst_population_score']:.2f}. The largest across-population score range occurs for **{largest_range['condition_label']}** at {largest_range['population_score_range']:.2f} points.

## Top institution by population

{top_lines}

## Public pool versus central clearing

For the cooperative population, the paired public-pool minus central-clearing difference is {coop_pair['mean_difference']:+.2f} points (95% CI {coop_pair['ci95_low']:+.2f} to {coop_pair['ci95_high']:+.2f}; n={int(coop_pair['n_paired_seeds'])}). Positive values favor the public pool.

## Build-mode comparison

Across institutions and populations, the unweighted mean crown-aware minus development-oriented score difference is {crown_average:+.2f}. The paper treats build mode as a secondary comparison rather than the main behavioral result.

The institutions are bundled protocol packages, so these comparisons do not identify the causal effect of a single institutional dimension.
"""
    (MARKDOWN_DIR / "fixed_institution_summary.md").write_text(memo, encoding="utf-8")


def generate_figures(
    mean_cells: pd.DataFrame,
    robustness: pd.DataFrame,
    institution_welfare: pd.DataFrame,
    paired_pool_central: pd.DataFrame,
    crown_effects: pd.DataFrame,
) -> pd.DataFrame:
    apply_report_style()
    manifest = [
        {
            "category": "main",
            "file": "plots/main/01_score_heatmap_development.png",
            "title": "Institutional performance depends on agent objectives",
            "purpose": "Full development-oriented institution × utility matrix",
        },
        {
            "category": "main",
            "file": "plots/main/02_robustness_average_worst_case.png",
            "title": "Average performance and robustness to changing populations",
            "purpose": "One point per institution: average versus worst population",
        },
        {
            "category": "main",
            "file": "plots/main/03_selected_institution_population_interaction.png",
            "title": "The best institutional arrangement changes with agent objectives",
            "purpose": "Selected population × institution interaction",
        },
        {
            "category": "main",
            "file": "plots/main/04_efficiency_weakest_agent_frontier.png",
            "title": "Aggregate development and the weakest-agent outcome",
            "purpose": "Aggregate score versus minimum agent score",
        },
        {
            "category": "main",
            "file": "plots/main/05_public_pool_vs_central_paired.png",
            "title": "When does contribution-based pooling outperform central matching?",
            "purpose": "Paired population-specific comparison with 95% intervals",
        },
        {
            "category": "supporting",
            "file": "plots/supporting/crown_score_effect_by_institution.png",
            "title": "Relative-status incentives are a secondary effect",
            "purpose": "Crown-aware minus development-oriented paired effects",
        },
    ]

    plot_development_score_heatmap(
        mean_cells, MAIN_PLOTS_DIR / "01_score_heatmap_development.png"
    )
    plot_robustness_frontier(
        robustness, MAIN_PLOTS_DIR / "02_robustness_average_worst_case.png"
    )
    plot_selected_interaction(
        mean_cells, MAIN_PLOTS_DIR / "03_selected_institution_population_interaction.png"
    )
    plot_efficiency_weakest_agent_frontier(
        institution_welfare, MAIN_PLOTS_DIR / "04_efficiency_weakest_agent_frontier.png"
    )
    plot_paired_public_pool_vs_central(
        paired_pool_central, MAIN_PLOTS_DIR / "05_public_pool_vs_central_paired.png"
    )
    plot_crown_effect_supporting(
        crown_effects, SUPPORTING_PLOTS_DIR / "crown_score_effect_by_institution.png"
    )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(EXPERIMENT_ROOT / "figure_manifest.csv", index=False)
    return manifest_df


def print_compact_diagnostics(
    n_seeds: int,
    summary: pd.DataFrame,
    robustness: pd.DataFrame,
    paired_pool_central: pd.DataFrame,
    rankings: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    print("Utility-results analysis complete")
    print(f"Source root: {EXPERIMENT_ROOT.resolve()}")
    print(f"Source summary rows: {len(summary):,}")
    print(f"Matched seeds per cell: {n_seeds}")
    print(f"Tables written: {len(list(TABLES_DIR.glob('*.csv')))}")
    print(f"Figures written: {len(manifest)}")

    print("\nTop institution by behavioral population:")
    for row in rankings[rankings["rank"] == 1].itertuples():
        print(
            f"- {row.population_label}: {row.condition_label} "
            f"(mean score={row.mean_final_total_score:.2f})"
        )

    print("\nRobustness across populations:")
    for row in robustness.sort_values("worst_population_score", ascending=False).itertuples():
        print(
            f"- {row.condition_label}: mean={row.mean_final_total_score:.2f}, "
            f"worst={row.worst_population_score:.2f}, "
            f"range={row.population_score_range:.2f}"
        )

    print("\nPublic pool minus central clearing, paired by seed:")
    for row in paired_pool_central.set_index("population").reindex(POPULATION_ORDER).reset_index().itertuples():
        print(
            f"- {row.population_label}: {row.mean_difference:+.2f} "
            f"[95% CI {row.ci95_low:+.2f}, {row.ci95_high:+.2f}]"
        )


def regenerate_utility_outputs() -> None:
    """Public entry point used by the experiment runner and standalone calls."""
    prepare_output_dirs()
    summary, agents, n_seeds = load_and_validate_sources()
    mean_cells = build_mean_cells(summary)
    agent_games, agent_cells = build_agent_seed_outcomes(agents)
    robustness = build_robustness_table(mean_cells)
    institution_welfare = build_institution_welfare_table(mean_cells, agent_cells)
    paired_pool_central = build_paired_institution_comparison(
        summary,
        first="public_pool",
        second="central_clearing",
    )
    crown_effects = build_crown_effects(summary)
    rankings = build_rankings(mean_cells)

    write_tables(
        mean_cells=mean_cells,
        agent_games=agent_games,
        agent_cells=agent_cells,
        robustness=robustness,
        institution_welfare=institution_welfare,
        paired_pool_central=paired_pool_central,
        crown_effects=crown_effects,
        rankings=rankings,
    )
    write_memo(
        n_seeds=n_seeds,
        summary=summary,
        mean_cells=mean_cells,
        robustness=robustness,
        paired_pool_central=paired_pool_central,
        crown_effects=crown_effects,
        rankings=rankings,
    )
    manifest = generate_figures(
        mean_cells=mean_cells,
        robustness=robustness,
        institution_welfare=institution_welfare,
        paired_pool_central=paired_pool_central,
        crown_effects=crown_effects,
    )
    print_compact_diagnostics(
        n_seeds=n_seeds,
        summary=summary,
        robustness=robustness,
        paired_pool_central=paired_pool_central,
        rankings=rankings,
        manifest=manifest,
    )


def main() -> None:
    regenerate_utility_outputs()


if __name__ == "__main__":
    main()
