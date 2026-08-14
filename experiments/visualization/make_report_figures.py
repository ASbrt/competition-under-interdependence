"""Generate the four main figures used by the paper from stored result tables."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "outputs" / "report_figures"
FIXED = ROOT / "experiments" / "outputs" / "fixed_institution"
COMPOSITION = ROOT / "experiments" / "outputs" / "composition_sweep"
PAPER_RESULTS = ROOT / "experiments" / "outputs" / "paper_results" / "tables"

POP_ORDER = [
    "Need-based",
    "Cooperative",
    "Selfish",
    "Hoarding",
    "Competitive",
    "Fairness-sensitive",
    "Mixed",
]
INST_ORDER = [
    "No trade",
    "Bilateral",
    "Bilateral 3-pass",
    "Catch-up",
    "Bottleneck",
    "Clearinghouse",
    "Subsidized",
    "Public pool",
    "Central capped (2)",
    "Equity central",
    "Central clearing",
]
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
    "central_cap2": "Central cap",
    "equity_cap2": "Equity cap",
    "central_full": "Full central",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def figure_1_fixed_institution() -> None:
    df = pd.read_csv(FIXED / "tables" / "development_cell_means.csv")
    matrix = (
        df.pivot(
            index="population_label",
            columns="condition_label",
            values="mean_final_total_score",
        )
        .reindex(index=POP_ORDER, columns=INST_ORDER)
    )
    data = matrix.to_numpy(float)
    fig, ax = plt.subplots(figsize=(13, 5.8))
    image = ax.imshow(data, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(INST_ORDER)), INST_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(POP_ORDER)), POP_ORDER)
    for row in range(data.shape[0]):
        best = int(np.nanargmax(data[row]))
        midpoint = float(np.nanmedian(data))
        for col in range(data.shape[1]):
            ax.text(
                col,
                row,
                f"{data[row, col]:.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if data[row, col] > midpoint else "black",
                fontweight="bold" if col == best else "normal",
            )
    ax.set_title("Fixed-institution benchmark: mean final total score")
    fig.colorbar(image, ax=ax, label="mean final total score")
    save(fig, "figure_1_fixed_institution")


def _composition_panel(ax: plt.Axes, grouped: pd.DataFrame, sweep: str, title: str) -> None:
    selected = [
        "Bilateral 3-pass",
        "Public pool",
        "Central capped (2)",
        "Central clearing",
    ]
    sub = grouped[grouped["sweep"] == sweep]
    for condition in selected:
        line = sub[sub["condition_label"] == condition].sort_values("restrictive_count")
        ax.errorbar(
            line["restrictive_count"],
            line["mean"],
            yerr=1.96 * line["se"],
            marker="o",
            linewidth=1.7,
            capsize=2,
            label=condition,
        )
    ax.set_title(title)
    ax.set_xlabel("agents using endpoint policy")
    ax.set_ylabel("mean final total score")
    ax.grid(axis="y", alpha=0.25)


def figure_2_composition_and_equity() -> None:
    composition = pd.read_csv(COMPOSITION / "csv" / "summary_by_seed.csv")
    grouped = (
        composition.groupby(
            ["sweep", "restrictive_count", "condition_label"],
            as_index=False,
        )
        .agg(
            mean=("final_total_score", "mean"),
            se=("final_total_score", lambda x: x.std(ddof=1) / sqrt(len(x))),
        )
    )

    fixed = pd.read_csv(FIXED / "tables" / "development_cell_means.csv")
    central = fixed[fixed["condition_label"].isin(["Central clearing", "Equity central"])].copy()
    score = (
        central.pivot(
            index="population_label",
            columns="condition_label",
            values="mean_final_total_score",
        )
        .reindex(POP_ORDER)
    )
    gap = (
        central.pivot(
            index="population_label",
            columns="condition_label",
            values="mean_final_score_gap",
        )
        .reindex(POP_ORDER)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    _composition_panel(
        axes[0, 0],
        grouped,
        "cooperative_to_hoarding",
        "Cooperative → hoarding",
    )
    _composition_panel(
        axes[0, 1],
        grouped,
        "need_based_to_competitive",
        "Need-based → competitive",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    y = np.arange(len(POP_ORDER))
    for ax, values, title, xlabel in [
        (axes[1, 0], score, "Aggregate development", "mean final total score"),
        (axes[1, 1], gap, "Distributional spread", "mean final max-min score range"),
    ]:
        ax.plot(values["Equity central"], y, "o", label="Equity central")
        ax.plot(values["Central clearing"], y, "o", label="Central clearing")
        for index in range(len(y)):
            ax.plot(
                [values.iloc[index]["Equity central"], values.iloc[index]["Central clearing"]],
                [y[index], y[index]],
                linewidth=1,
                alpha=0.35,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
    axes[1, 0].set_yticks(y, POP_ORDER)
    axes[1, 0].invert_yaxis()
    axes[1, 1].set_yticks(y, [])
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Behavioral composition and the central–equity comparison")
    fig.tight_layout()
    save(fig, "figure_2_composition_and_equity")


def figure_3_action_allocation() -> None:
    df = pd.read_csv(PAPER_RESULTS / "action_shares_pooled.csv")
    q = df[df["policy_name"] == "learned_q"].set_index("action").reindex(ACTION_ORDER)
    shares = q["share"].to_numpy(float) * 100
    labels = [ACTION_LABELS[action] for action in ACTION_ORDER]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(y, shares)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("share of evaluation decisions (%)")
    ax.set_title("Learned Q-policy: realized institutional action allocation")
    for yi, value in zip(y, shares):
        ax.text(value + 0.5, yi, f"{value:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "figure_3_action_allocation")


def figure_4_policy_comparison() -> None:
    df = pd.read_csv(PAPER_RESULTS / "policy_comparisons.csv")
    order = [
        ("permanent_bilateral_3pass", "Permanent bilateral"),
        ("random_feasible", "Random feasible"),
        ("frequency_informed_random", "Frequency-informed random"),
        ("shuffled_learned_sequence", "Shuffled learned sequence"),
    ]
    rows = []
    for policy, label in order:
        row = df[
            (df["target_policy"] == "learned_q")
            & (df["baseline_policy"] == policy)
            & (df["metric"] == "final_welfare")
        ].iloc[0]
        rows.append((label, row))

    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axvline(0, linewidth=1)
    for yi, (label, row) in zip(y, rows):
        value = float(row["mean_difference"])
        low = float(row["seed_clustered_ci95_low"])
        high = float(row["seed_clustered_ci95_high"])
        ax.plot([low, high], [yi, yi], linewidth=3)
        ax.scatter(value, yi, s=55, zorder=3)
        ax.text(value, yi + 0.16, f"{value:+.2f}", ha="center", fontsize=9)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.set_xlabel("learned Q minus comparison policy: final planner welfare")
    ax.set_title("Paired adaptive-policy comparisons")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "figure_4_policy_comparison")


def main() -> None:
    figure_1_fixed_institution()
    figure_2_composition_and_equity()
    figure_3_action_allocation()
    figure_4_policy_comparison()
    print(f"wrote four report figures to {OUT}")


if __name__ == "__main__":
    main()
