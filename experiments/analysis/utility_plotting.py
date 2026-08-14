"""Plotting helpers for the fixed-institution benchmark.

This module consumes derived data frames only and never runs the simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

REPORT_DPI = 240

POPULATION_ORDER = [
    "need_based",
    "cooperative",
    "selfish",
    "hoarding",
    "competitive",
    "fairness_sensitive",
    "mixed",
]
POPULATION_LABELS = {
    "need_based": "Need-based",
    "cooperative": "Cooperative",
    "selfish": "Selfish",
    "hoarding": "Hoarding",
    "competitive": "Competitive",
    "fairness_sensitive": "Fairness-sensitive",
    "mixed": "Mixed",
}

INSTITUTION_ORDER = [
    "no_trade",
    "bilateral_trade",
    "bilateral_trade_3pass",
    "catch_up_bilateral_trade",
    "bottleneck_priority_bilateral_trade",
    "clearinghouse_bargaining",
    "subsidized_catch_up",
    "public_pool",
    "central_clearing",
    "equity_weighted_central",
    "central_clearing_capped",
]
INSTITUTION_LABELS = {
    "no_trade": "No trade",
    "bilateral_trade": "Bilateral",
    "bilateral_trade_3pass": "Bilateral 3-pass",
    "catch_up_bilateral_trade": "Catch-up",
    "bottleneck_priority_bilateral_trade": "Bottleneck",
    "clearinghouse_bargaining": "Clearinghouse",
    "subsidized_catch_up": "Subsidized",
    "public_pool": "Public pool",
    "central_clearing": "Central clearing",
    "equity_weighted_central": "Equity central",
    "central_clearing_capped": "Central capped (2)",
}

SELECTED_POPULATIONS = ["need_based", "cooperative", "hoarding", "mixed"]
SELECTED_INSTITUTIONS = [
    "no_trade",
    "bilateral_trade_3pass",
    "public_pool",
    "central_clearing_capped",
    "central_clearing",
    "equity_weighted_central",
]


def apply_report_style() -> None:
    """Apply a clean, consistent style suitable for slides and a paper."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 17,
            "figure.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: Figure, path: Path, *, pad_inches: float = 0.18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=REPORT_DPI,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    plt.close(fig)


def _direct_label_points(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    label: str,
) -> None:
    """Add compact labels next to points with deterministic alternating offsets."""
    offsets = [(6, 5), (6, -9), (-6, 6), (-6, -10)]
    for index, (_, row) in enumerate(data.iterrows()):
        dx, dy = offsets[index % len(offsets)]
        ax.annotate(
            str(row[label]),
            (row[x], row[y]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=8.5,
        )


def plot_development_score_heatmap(
    mean_cells: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the full development-oriented institution × utility matrix."""
    frame = (
        mean_cells.pivot(
            index="population",
            columns="condition",
            values="mean_final_total_score",
        )
        .reindex(index=POPULATION_ORDER, columns=INSTITUTION_ORDER)
        .rename(index=POPULATION_LABELS, columns=INSTITUTION_LABELS)
    )

    fig, ax = plt.subplots(figsize=(16.5, 7.4), layout="constrained")
    sns.heatmap(
        frame,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.55,
        linecolor="white",
        cbar_kws={"label": "Mean final total score", "shrink": 0.88},
        ax=ax,
    )
    ax.set_title("Institutional performance depends on agent objectives")
    ax.set_xlabel("Exchange institution")
    ax.set_ylabel("Utility population")
    ax.tick_params(axis="x", rotation=29)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.text(
        0.5,
        -0.02,
        "Development-oriented specification; cell values are means across matched simulation seeds.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output_path)


def plot_robustness_frontier(
    robustness: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mean performance against worst observed population performance."""
    ordered = robustness.sort_values("mean_final_total_score").copy()
    fig, ax = plt.subplots(figsize=(11.5, 7.2), layout="constrained")
    sns.scatterplot(
        data=ordered,
        x="mean_final_total_score",
        y="worst_population_score",
        hue="institution_family_label",
        style="institution_family_label",
        s=115,
        ax=ax,
    )
    _direct_label_points(
        ax,
        ordered,
        x="mean_final_total_score",
        y="worst_population_score",
        label="condition_label",
    )
    ax.set_title("Average performance and robustness to changing populations")
    ax.set_xlabel("Mean final total score across utility populations")
    ax.set_ylabel("Worst population-specific mean score")
    ax.legend(title="Institutional logic", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.text(
        0.5,
        -0.02,
        "Institutions farther toward the upper right combine stronger average and worst-case performance.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output_path)


def plot_selected_interaction(
    mean_cells: pd.DataFrame,
    output_path: Path,
) -> None:
    """Show a compact set of institution-by-population interactions."""
    selected = mean_cells[
        mean_cells["population"].isin(SELECTED_POPULATIONS)
        & mean_cells["condition"].isin(SELECTED_INSTITUTIONS)
    ].copy()
    selected["condition"] = pd.Categorical(
        selected["condition"], categories=SELECTED_INSTITUTIONS, ordered=True
    )
    selected["population"] = pd.Categorical(
        selected["population"], categories=SELECTED_POPULATIONS, ordered=True
    )
    selected = selected.sort_values(["population", "condition"])
    selected["condition_label"] = selected["condition"].astype(str).map(INSTITUTION_LABELS)
    selected["population_label"] = selected["population"].astype(str).map(POPULATION_LABELS)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout="constrained")
    sns.lineplot(
        data=selected,
        x="condition_label",
        y="mean_final_total_score",
        hue="population_label",
        style="population_label",
        markers=True,
        dashes=False,
        linewidth=2.0,
        markersize=8,
        ax=ax,
    )
    ax.set_title("The best institutional arrangement changes with agent objectives")
    ax.set_xlabel("Institution")
    ax.set_ylabel("Mean final total score")
    ax.tick_params(axis="x", rotation=24)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(title="Utility population", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_figure(fig, output_path)


def plot_efficiency_weakest_agent_frontier(
    institution_welfare: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot aggregate performance against the weakest-agent outcome."""
    ordered = institution_welfare.sort_values("mean_final_total_score").copy()
    fig, ax = plt.subplots(figsize=(11.5, 7.2), layout="constrained")
    sns.scatterplot(
        data=ordered,
        x="mean_min_agent_score",
        y="mean_final_total_score",
        hue="institution_family_label",
        style="institution_family_label",
        s=115,
        ax=ax,
    )
    _direct_label_points(
        ax,
        ordered,
        x="mean_min_agent_score",
        y="mean_final_total_score",
        label="condition_label",
    )
    ax.set_title("Aggregate development and the weakest-agent outcome")
    ax.set_xlabel("Mean final score of the lowest-scoring agent")
    ax.set_ylabel("Mean final total score")
    ax.legend(title="Institutional logic", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.text(
        0.5,
        -0.02,
        "Each point averages across utility populations; upper-right institutions perform well on both dimensions.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output_path)


def plot_paired_public_pool_vs_central(
    paired_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot seed-paired public-pool minus central-clearing score differences."""
    ordered = paired_comparison.copy()
    ordered["population"] = pd.Categorical(
        ordered["population"], categories=POPULATION_ORDER[::-1], ordered=True
    )
    ordered = ordered.sort_values("population")
    y = np.arange(len(ordered))
    means = ordered["mean_difference"].to_numpy()
    lower = means - ordered["ci95_low"].to_numpy()
    upper = ordered["ci95_high"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(10.6, 6.8), layout="constrained")
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=7,
    )
    ax.axvline(0, linewidth=1.1, color="#555555")
    ax.set_yticks(y, [POPULATION_LABELS[value] for value in ordered["population"].astype(str)])
    ax.set_xlabel("Paired score difference: public pool − central clearing")
    ax.set_ylabel("Utility population")
    ax.set_title("When does contribution-based pooling outperform central matching?")
    fig.text(
        0.5,
        -0.02,
        "Positive values favor the public pool; intervals are based on matched simulation seeds.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output_path)


def plot_crown_effect_supporting(
    crown_effects: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the secondary crown-aware score effect by institution."""
    ordered = crown_effects.set_index("condition").reindex(INSTITUTION_ORDER).reset_index()
    y = np.arange(len(ordered))
    means = ordered["mean_difference"].to_numpy()
    lower = means - ordered["ci95_low"].to_numpy()
    upper = ordered["ci95_high"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(10.8, 7.2), layout="constrained")
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        markersize=6,
    )
    ax.axvline(0, linewidth=1.0, color="#555555")
    ax.set_yticks(y, ordered["condition_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Paired score difference: crown-aware − development-oriented")
    ax.set_ylabel("Institution")
    ax.set_title("Relative-status incentives are a secondary effect in the revised model")
    save_figure(fig, output_path)
