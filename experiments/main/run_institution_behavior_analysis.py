"""Run the fixed-institution benchmark used in the paper.

The experiment crosses two build modes, seven behavioral populations, eleven
exchange institutions, and matched simulation seeds.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs"
INTERNAL_OUTPUT_ROOT = EXPERIMENTS_OUTPUT_ROOT / ".internal"
MPLCONFIGDIR = INTERNAL_OUTPUT_ROOT / ".mplconfig"
XDG_CACHE_HOME = INTERNAL_OUTPUT_ROOT / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires pandas, seaborn, and matplotlib.\n\n"
        "Install them with:\n"
        "pip install pandas seaborn matplotlib"
    ) from exc

from engine.agents import Agent
from engine.build_rules import BuildRules
from engine.game import Game
from engine.institutions import (
    BilateralTradeInstitution,
    BottleneckPriorityBilateralTradeInstitution,
    CatchUpBilateralTradeInstitution,
    CentralMarketClearingInstitution,
    ClearinghouseBargainingInstitution,
    EquityWeightedCentralClearingInstitution,
    NoTradeInstitution,
    PublicPoolInstitution,
    SubsidizedCatchUpInstitution,
)
from engine.policies import (
    BUILD_MODE_CROWN_AWARE,
    BUILD_MODE_DEVELOPMENT_ORIENTED,
    AgentPolicy,
    CompetitiveTradePolicy,
    CooperativeTradePolicy,
    FairnessSensitiveTradePolicy,
    HoardingTradePolicy,
    NeedBasedTradePolicy,
    SelfishTradePolicy,
)
from engine.resources import (
    RESOURCES,
    ResourceAccessProfile,
    create_random_balanced_access_profiles,
)


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer environment variable with a useful error."""
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


BUILD_MODES = [
    BUILD_MODE_DEVELOPMENT_ORIENTED,
    BUILD_MODE_CROWN_AWARE,
]
N_SEEDS = _int_env("N_SEEDS", 200)
ROUNDS = _int_env("ROUNDS", 20)
MAX_BUILDS_PER_AGENT_PER_ROUND: int | None = _int_env(
    "MAX_BUILDS_PER_AGENT_PER_ROUND",
    4,
)

OUTPUT_ROOT = EXPERIMENTS_OUTPUT_ROOT / "fixed_institution"
CSV_DIR = OUTPUT_ROOT / "csv"
PLOTS_DIR = OUTPUT_ROOT / "plots"

BUILD_MODE_LABELS = {
    BUILD_MODE_DEVELOPMENT_ORIENTED: "Development-oriented",
    BUILD_MODE_CROWN_AWARE: "Crown-aware",
}
BUILD_MODE_LABEL_ORDER = [BUILD_MODE_LABELS[key] for key in BUILD_MODES]

RESOURCE_SHORT_LABELS = {
    "materials": "Mat",
    "components": "Comp",
    "food": "Food",
    "energy": "Energy",
    "knowledge": "Know",
}

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
INSTITUTION_LABEL_ORDER = [INSTITUTION_LABELS[key] for key in INSTITUTION_ORDER]

POPULATION_LABELS = {
    "need_based": "Need-based",
    "cooperative": "Cooperative",
    "selfish": "Selfish",
    "hoarding": "Hoarding",
    "competitive": "Competitive",
    "fairness_sensitive": "Fairness-sensitive",
    "mixed": "Mixed",
}
POPULATION_ORDER = [
    "need_based",
    "cooperative",
    "selfish",
    "hoarding",
    "competitive",
    "fairness_sensitive",
    "mixed",
]
POPULATION_LABEL_ORDER = [POPULATION_LABELS[key] for key in POPULATION_ORDER]

CORE_SUMMARY_METRICS = [
    "final_total_score",
    "final_score_gap",
    "cumulative_trades_executed",
    "final_idle_infrastructure",
    "final_total_infrastructure",
    "final_total_production_sites",
    "final_total_advanced_sites",
    "final_total_innovation",
]

MIXED_POLICY_CLASSES: tuple[type[AgentPolicy], ...] = (
    CooperativeTradePolicy,
    SelfishTradePolicy,
    HoardingTradePolicy,
    CompetitiveTradePolicy,
    FairnessSensitiveTradePolicy,
)

RNG_STREAM_ACCESS = 0
RNG_STREAM_PRODUCTION = 1
RNG_STREAM_INSTITUTION = 2
RNG_STREAM_MIXED_ASSIGNMENT = 3


def make_rng(seed: int, stream_id: int) -> np.random.Generator:
    """Create a reproducible random stream independent of other mechanisms."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream_id)]))


def shuffled_mixed_policy_order(seed: int) -> tuple[type[AgentPolicy], ...]:
    """Randomize behavior-to-agent assignment independently of access profiles.

    The returned order is deterministic for a seed and is reused across all
    institutions and build modes, preserving paired comparisons without fixing
    a behavioral type to a particular primary-resource position.
    """
    order = list(MIXED_POLICY_CLASSES)
    make_rng(seed, RNG_STREAM_MIXED_ASSIGNMENT).shuffle(order)
    return tuple(order)


def infer_primary_secondary_resources(
    profile: ResourceAccessProfile,
) -> tuple[str, str]:
    """Read explicit access labels or infer them from the two largest weights."""
    if profile.primary_resource is not None and profile.secondary_resource is not None:
        return profile.primary_resource, profile.secondary_resource

    ranked_resources = sorted(
        RESOURCES,
        key=lambda resource: (-profile.probabilities[resource], RESOURCES.index(resource)),
    )
    return ranked_resources[0], ranked_resources[1]


def access_pair_label_from_profile(profile: ResourceAccessProfile) -> str:
    """Format an access profile as a compact primary-to-secondary label."""
    primary_resource, secondary_resource = infer_primary_secondary_resources(profile)
    return (
        f"{RESOURCE_SHORT_LABELS.get(primary_resource, primary_resource)} -> "
        f"{RESOURCE_SHORT_LABELS.get(secondary_resource, secondary_resource)}"
    )


def prepare_output_dirs() -> None:
    """Recreate the output folders for a full fixed-institution run."""
    generated_paths = [
        CSV_DIR,
        OUTPUT_ROOT / "tables",
        OUTPUT_ROOT / "markdown",
        PLOTS_DIR / "main",
        PLOTS_DIR / "supporting",
    ]
    for path in generated_paths:
        if path.exists():
            shutil.rmtree(path)
    CSV_DIR.mkdir(parents=True, exist_ok=True)


def create_agents(profiles: list[ResourceAccessProfile]) -> list[Agent]:
    """Create one fresh agent per access profile in stable profile order."""
    return [
        Agent(id=index, access_profile=profile)
        for index, profile in enumerate(profiles)
    ]


def build_population_specs() -> list[dict[str, Any]]:
    """Define homogeneous and mixed policy populations used in the factorial."""
    return [
        {
            "name": "need_based",
            "label": POPULATION_LABELS["need_based"],
            "policy_builder": lambda agents, build_mode: {
                agent.id: NeedBasedTradePolicy(build_mode=build_mode)
                for agent in agents
            },
            "policy_population_type": "homogeneous_need_based",
        },
        {
            "name": "cooperative",
            "label": POPULATION_LABELS["cooperative"],
            "policy_builder": lambda agents, build_mode: {
                agent.id: CooperativeTradePolicy(build_mode=build_mode)
                for agent in agents
            },
            "policy_population_type": "homogeneous_cooperative",
        },
        {
            "name": "selfish",
            "label": POPULATION_LABELS["selfish"],
            "policy_builder": lambda agents, build_mode: {
                agent.id: SelfishTradePolicy(build_mode=build_mode)
                for agent in agents
            },
            "policy_population_type": "homogeneous_selfish",
        },
        {
            "name": "hoarding",
            "label": POPULATION_LABELS["hoarding"],
            "policy_builder": lambda agents, build_mode: {
                agent.id: HoardingTradePolicy(build_mode=build_mode)
                for agent in agents
            },
            "policy_population_type": "homogeneous_hoarding",
        },
        {
            "name": "competitive",
            "label": POPULATION_LABELS["competitive"],
            "policy_builder": lambda agents, build_mode: {
                agent.id: CompetitiveTradePolicy(build_mode=build_mode)
                for agent in agents
            },
            "policy_population_type": "homogeneous_competitive",
        },
        {
            "name": "fairness_sensitive",
            "label": POPULATION_LABELS["fairness_sensitive"],
            "policy_builder": lambda agents, build_mode: {
                agent.id: FairnessSensitiveTradePolicy(build_mode=build_mode)
                for agent in agents
            },
            "policy_population_type": "homogeneous_fairness_sensitive",
        },
        {
            "name": "mixed",
            "label": POPULATION_LABELS["mixed"],
            "policy_builder": None,
            "policy_population_type": "mixed_shuffled_within_seed",
        },
    ]


def build_institution_specs() -> list[dict[str, Any]]:
    """Define institution factories, labels, and exported configuration metadata."""
    return [
        {
            "name": "no_trade",
            "label": INSTITUTION_LABELS["no_trade"],
            "institution_factory": NoTradeInstitution,
            "institution_type": "no_trade",
            "max_bargaining_passes": None,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "bilateral_trade",
            "label": INSTITUTION_LABELS["bilateral_trade"],
            "institution_factory": lambda: BilateralTradeInstitution(max_bargaining_passes=1),
            "institution_type": "bilateral",
            "max_bargaining_passes": 1,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "bilateral_trade_3pass",
            "label": INSTITUTION_LABELS["bilateral_trade_3pass"],
            "institution_factory": lambda: BilateralTradeInstitution(max_bargaining_passes=3),
            "institution_type": "bilateral",
            "max_bargaining_passes": 3,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "catch_up_bilateral_trade",
            "label": INSTITUTION_LABELS["catch_up_bilateral_trade"],
            "institution_factory": lambda: CatchUpBilateralTradeInstitution(max_bargaining_passes=1),
            "institution_type": "bilateral_priority",
            "max_bargaining_passes": 1,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "bottleneck_priority_bilateral_trade",
            "label": INSTITUTION_LABELS["bottleneck_priority_bilateral_trade"],
            "institution_factory": lambda: BottleneckPriorityBilateralTradeInstitution(max_bargaining_passes=1),
            "institution_type": "bilateral_priority",
            "max_bargaining_passes": 1,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "clearinghouse_bargaining",
            "label": INSTITUTION_LABELS["clearinghouse_bargaining"],
            "institution_factory": lambda: ClearinghouseBargainingInstitution(
                max_bargaining_passes=1,
                max_trades_per_round=None,
            ),
            "institution_type": "clearinghouse_bargaining",
            "max_bargaining_passes": 1,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "subsidized_catch_up",
            "label": INSTITUTION_LABELS["subsidized_catch_up"],
            "institution_factory": SubsidizedCatchUpInstitution,
            "institution_type": "redistributive_support",
            "max_bargaining_passes": None,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "public_pool",
            "label": INSTITUTION_LABELS["public_pool"],
            "institution_factory": lambda: PublicPoolInstitution(
                max_allocations_per_round=3,
                prioritize_low_score=True,
            ),
            "institution_type": "public_pool",
            "max_bargaining_passes": None,
            "max_trades_per_round": None,
            "max_allocations_per_round": 3,
            "equity_weight": None,
        },
        {
            "name": "central_clearing",
            "label": INSTITUTION_LABELS["central_clearing"],
            "institution_factory": lambda: CentralMarketClearingInstitution(max_trades_per_round=None),
            "institution_type": "central_clearing",
            "max_bargaining_passes": None,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
        {
            "name": "equity_weighted_central",
            "label": INSTITUTION_LABELS["equity_weighted_central"],
            "institution_factory": lambda: EquityWeightedCentralClearingInstitution(
                equity_weight=1.0,
                max_trades_per_round=None,
            ),
            "institution_type": "equity_weighted_central",
            "max_bargaining_passes": None,
            "max_trades_per_round": None,
            "max_allocations_per_round": None,
            "equity_weight": 1.0,
        },
        {
            "name": "central_clearing_capped",
            "label": INSTITUTION_LABELS["central_clearing_capped"],
            "institution_factory": lambda: CentralMarketClearingInstitution(max_trades_per_round=2),
            "institution_type": "central_clearing",
            "max_bargaining_passes": None,
            "max_trades_per_round": 2,
            "max_allocations_per_round": None,
            "equity_weight": None,
        },
    ]


def build_policies_for_population(
    population_spec: dict[str, Any],
    agents: list[Agent],
    build_mode: str,
    mixed_policy_order: tuple[type[AgentPolicy], ...] | None = None,
) -> dict[int, AgentPolicy]:
    """Construct fresh policy instances for every agent in one game.

    Homogeneous populations use their ordinary builder. The mixed population
    receives a seed-specific shuffled policy order that is held fixed across
    all institutions and build modes for matched comparisons.
    """
    if population_spec["name"] != "mixed":
        builder = population_spec["policy_builder"]
        if builder is None:
            raise ValueError("Non-mixed population is missing a policy builder.")
        return builder(agents, build_mode)

    if mixed_policy_order is None:
        raise ValueError("mixed_policy_order is required for the mixed population.")
    if len(mixed_policy_order) != len(agents):
        raise ValueError(
            "mixed_policy_order must contain exactly one policy class per agent."
        )

    return {
        agent.id: mixed_policy_order[index](build_mode=build_mode)
        for index, agent in enumerate(agents)
    }


def summarize_finished_game(
    game: Game,
    build_mode: str,
    population_spec: dict[str, Any],
    institution_spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Collapse one completed game to a single seed-level result row."""
    final_metrics = game.history[-1]

    summary = {
        "build_mode": build_mode,
        "build_mode_label": BUILD_MODE_LABELS[build_mode],
        "population": population_spec["name"],
        "population_label": population_spec["label"],
        "condition": institution_spec["name"],
        "condition_label": institution_spec["label"],
        "seed": seed,
        "institution_type": institution_spec["institution_type"],
        "policy_population_type": population_spec["policy_population_type"],
        "max_bargaining_passes": institution_spec["max_bargaining_passes"],
        "max_trades_per_round": institution_spec["max_trades_per_round"],
        "max_allocations_per_round": institution_spec["max_allocations_per_round"],
        "equity_weight": institution_spec["equity_weight"],
        "max_builds_per_agent_per_round": game.max_builds_per_agent_per_round,
        "final_total_score": final_metrics["total_score"],
        "final_total_resources": final_metrics["total_resources"],
        "final_score_gap": final_metrics["score_gap"],
        "final_total_infrastructure": final_metrics["total_infrastructure"],
        "final_idle_infrastructure": final_metrics["idle_infrastructure"],
        "final_total_production_sites": final_metrics["total_production_sites"],
        "final_total_advanced_sites": final_metrics["total_advanced_sites"],
        "final_total_innovation": final_metrics["total_innovation"],
        "cumulative_trades_proposed": sum(
            round_metrics["trades_proposed"]
            for round_metrics in game.history
        ),
        "cumulative_trades_executed": sum(
            round_metrics["trades_executed"]
            for round_metrics in game.history
        ),
        "infrastructure_leader_id": final_metrics["infrastructure_leader_id"],
        "innovation_leader_id": final_metrics["innovation_leader_id"],
    }

    for resource in RESOURCES:
        summary[f"final_total_{resource}"] = final_metrics[f"total_{resource}"]

    return summary


def collect_round_history(
    game: Game,
    build_mode: str,
    population_spec: dict[str, Any],
    institution_spec: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Attach experiment identifiers to every stored round metric."""
    rows = []
    for round_metrics in game.history:
        rows.append(
            {
                "build_mode": build_mode,
                "build_mode_label": BUILD_MODE_LABELS[build_mode],
                "population": population_spec["name"],
                "population_label": population_spec["label"],
                "condition": institution_spec["name"],
                "condition_label": institution_spec["label"],
                "seed": seed,
                "institution_type": institution_spec["institution_type"],
                "policy_population_type": population_spec["policy_population_type"],
                "max_bargaining_passes": institution_spec["max_bargaining_passes"],
                "max_trades_per_round": institution_spec["max_trades_per_round"],
                "max_allocations_per_round": institution_spec["max_allocations_per_round"],
                "equity_weight": institution_spec["equity_weight"],
                "max_builds_per_agent_per_round": game.max_builds_per_agent_per_round,
                **round_metrics,
            }
        )

    return rows


def collect_final_agents(
    game: Game,
    policies: dict[int, AgentPolicy],
    build_mode: str,
    population_spec: dict[str, Any],
    institution_spec: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Export one final-state row per agent, including tie-adjusted wins."""
    final_scores = [agent.score for agent in game.agents]
    winning_score = max(final_scores)
    winner_ids = [
        agent.id
        for agent in game.agents
        if agent.score == winning_score
    ]
    winner_credit = 1.0 / len(winner_ids)

    rows = []
    for agent in game.agents:
        primary_resource, secondary_resource = infer_primary_secondary_resources(
            agent.access_profile
        )
        rows.append(
            {
                "build_mode": build_mode,
                "build_mode_label": BUILD_MODE_LABELS[build_mode],
                "population": population_spec["name"],
                "population_label": population_spec["label"],
                "condition": institution_spec["name"],
                "condition_label": institution_spec["label"],
                "seed": seed,
                "agent_id": agent.id,
                "institution_type": institution_spec["institution_type"],
                "policy_population_type": population_spec["policy_population_type"],
                "max_bargaining_passes": institution_spec["max_bargaining_passes"],
                "max_trades_per_round": institution_spec["max_trades_per_round"],
                "max_allocations_per_round": institution_spec["max_allocations_per_round"],
                "equity_weight": institution_spec["equity_weight"],
                "max_builds_per_agent_per_round": game.max_builds_per_agent_per_round,
                "policy_class_name": policies[agent.id].__class__.__name__,
                "access_profile_name": agent.access_profile.name,
                "access_pair_label": access_pair_label_from_profile(agent.access_profile),
                "primary_resource": primary_resource,
                "secondary_resource": secondary_resource,
                "score": agent.score,
                "infrastructure": agent.infrastructure,
                "production_sites": agent.production_sites,
                "advanced_sites": agent.advanced_sites,
                "innovation": agent.innovation,
                "stock_materials": agent.stock["materials"],
                "stock_components": agent.stock["components"],
                "stock_food": agent.stock["food"],
                "stock_energy": agent.stock["energy"],
                "stock_knowledge": agent.stock["knowledge"],
                "is_winner": agent.id in winner_ids,
                "winner_credit": winner_credit if agent.id in winner_ids else 0.0,
            }
        )

    return rows


def run_all_experiments(
    seeds,
    rounds: int,
    max_builds_per_agent_per_round: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the matched-seed factorial and return raw result tables.

    Four independent streams are used per seed: access-profile generation,
    production, institution tie-breaking, and mixed-population assignment. The
    production stream is reset identically for every condition, so different
    institutions cannot alter future exogenous production draws merely by
    consuming different numbers of random tie-breaks. The shuffled mixed-policy
    assignment is reused across institutions and build modes within a seed.
    """
    population_specs = build_population_specs()
    institution_specs = build_institution_specs()

    summary_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    final_agent_rows: list[dict[str, Any]] = []

    for seed in seeds:
        profile_rng = make_rng(seed, RNG_STREAM_ACCESS)
        profiles = list(create_random_balanced_access_profiles(profile_rng).values())
        mixed_order = shuffled_mixed_policy_order(seed)

        for build_mode in BUILD_MODES:
            for population_spec in population_specs:
                for institution_spec in institution_specs:
                    institution = institution_spec["institution_factory"]()
                    agents = create_agents(profiles)
                    policies = build_policies_for_population(
                        population_spec=population_spec,
                        agents=agents,
                        build_mode=build_mode,
                        mixed_policy_order=(
                            mixed_order if population_spec["name"] == "mixed" else None
                        ),
                    )
                    game = Game(
                        agents=agents,
                        policies=policies,
                        build_rules=BuildRules(),
                        institution=institution,
                        rng=make_rng(seed, RNG_STREAM_INSTITUTION),
                        production_rng=make_rng(seed, RNG_STREAM_PRODUCTION),
                        max_builds_per_agent_per_round=max_builds_per_agent_per_round,
                    )

                    for _ in range(rounds):
                        game.step()

                    summary_rows.append(
                        summarize_finished_game(
                            game=game,
                            build_mode=build_mode,
                            population_spec=population_spec,
                            institution_spec=institution_spec,
                            seed=seed,
                        )
                    )
                    round_rows.extend(
                        collect_round_history(
                            game=game,
                            build_mode=build_mode,
                            population_spec=population_spec,
                            institution_spec=institution_spec,
                            seed=seed,
                        )
                    )
                    final_agent_rows.extend(
                        collect_final_agents(
                            game=game,
                            policies=game.policies,
                            build_mode=build_mode,
                            population_spec=population_spec,
                            institution_spec=institution_spec,
                            seed=seed,
                        )
                    )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(round_rows),
        pd.DataFrame(final_agent_rows),
    )


def _safe_percent_change(delta: pd.Series, baseline: pd.Series) -> pd.Series:
    """Calculate percent change while leaving zero baselines undefined."""
    denominator = baseline.replace(0, np.nan).abs()
    return 100.0 * delta / denominator


def save_summary_csvs(
    summary_df: pd.DataFrame,
    round_history_df: pd.DataFrame,
    final_agents_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write raw and derived CSVs and return the derived analysis tables.

    Build-mode effects are paired cell means defined as crown-aware minus
    development-oriented. Rankings are descriptive orderings of those means.
    """
    summary_df.to_csv(CSV_DIR / "summary_by_seed.csv", index=False)
    round_history_df.to_csv(CSV_DIR / "round_history.csv", index=False)
    final_agents_df.to_csv(CSV_DIR / "final_agents.csv", index=False)

    mean_by_build_population_condition = (
        summary_df.groupby(
            [
                "build_mode",
                "build_mode_label",
                "population",
                "population_label",
                "condition",
                "condition_label",
            ],
            observed=False,
        )[CORE_SUMMARY_METRICS]
        .mean()
        .reset_index()
    )
    mean_by_build_population_condition.to_csv(
        CSV_DIR / "mean_by_build_population_condition.csv",
        index=False,
    )

    dev_df = mean_by_build_population_condition[
        mean_by_build_population_condition["build_mode"] == BUILD_MODE_DEVELOPMENT_ORIENTED
    ].copy()
    crown_df = mean_by_build_population_condition[
        mean_by_build_population_condition["build_mode"] == BUILD_MODE_CROWN_AWARE
    ].copy()

    merge_keys = [
        "population",
        "population_label",
        "condition",
        "condition_label",
    ]
    effect_df = dev_df.merge(
        crown_df,
        on=merge_keys,
        suffixes=("_development", "_crown"),
    )
    effect_df["build_mode_comparison"] = "crown_aware_minus_development_oriented"
    effect_df["build_mode_from"] = BUILD_MODE_DEVELOPMENT_ORIENTED
    effect_df["build_mode_to"] = BUILD_MODE_CROWN_AWARE

    delta_map = {
        "final_total_score": "delta_final_total_score",
        "final_score_gap": "delta_final_score_gap",
        "cumulative_trades_executed": "delta_cumulative_trades_executed",
        "final_idle_infrastructure": "delta_final_idle_infrastructure",
        "final_total_infrastructure": "delta_final_total_infrastructure",
        "final_total_production_sites": "delta_final_total_production_sites",
        "final_total_advanced_sites": "delta_final_total_advanced_sites",
        "final_total_innovation": "delta_final_total_innovation",
    }
    for base_name, delta_name in delta_map.items():
        effect_df[delta_name] = (
            effect_df[f"{base_name}_crown"] - effect_df[f"{base_name}_development"]
        )

    effect_df["pct_change_final_total_score"] = _safe_percent_change(
        effect_df["delta_final_total_score"],
        effect_df["final_total_score_development"],
    )
    effect_df["pct_change_final_score_gap"] = _safe_percent_change(
        effect_df["delta_final_score_gap"],
        effect_df["final_score_gap_development"],
    )
    effect_df["pct_change_cumulative_trades_executed"] = _safe_percent_change(
        effect_df["delta_cumulative_trades_executed"],
        effect_df["cumulative_trades_executed_development"],
    )
    effect_df.to_csv(
        CSV_DIR / "build_mode_effect_by_population_condition.csv",
        index=False,
    )

    rankings_rows: list[dict[str, Any]] = []
    for (build_mode, population), group in mean_by_build_population_condition.groupby(
        ["build_mode", "population"],
        observed=False,
    ):
        score_ranked = group.sort_values(
            ["final_total_score", "condition_label"],
            ascending=[False, True],
        ).reset_index(drop=True)
        gap_ranked = group.sort_values(
            ["final_score_gap", "condition_label"],
            ascending=[True, True],
        ).reset_index(drop=True)
        score_ranks = {
            condition: rank + 1
            for rank, condition in enumerate(score_ranked["condition"])
        }
        gap_ranks = {
            condition: rank + 1
            for rank, condition in enumerate(gap_ranked["condition"])
        }
        for _, row in group.iterrows():
            rankings_rows.append(
                {
                    "build_mode": row["build_mode"],
                    "build_mode_label": row["build_mode_label"],
                    "population": row["population"],
                    "population_label": row["population_label"],
                    "condition": row["condition"],
                    "condition_label": row["condition_label"],
                    "mean_final_total_score": row["final_total_score"],
                    "mean_final_score_gap": row["final_score_gap"],
                    "rank_final_total_score": score_ranks[row["condition"]],
                    "rank_lowest_final_score_gap": gap_ranks[row["condition"]],
                }
            )

    rankings_df = pd.DataFrame(rankings_rows)
    rankings_df.to_csv(
        CSV_DIR / "institution_rankings_by_build_population.csv",
        index=False,
    )

    population_sensitivity_df = (
        mean_by_build_population_condition.groupby(
            ["build_mode", "build_mode_label", "condition", "condition_label"],
            observed=False,
        )
        .agg(
            mean_final_total_score=("final_total_score", "mean"),
            min_final_total_score=("final_total_score", "min"),
            max_final_total_score=("final_total_score", "max"),
            mean_final_score_gap=("final_score_gap", "mean"),
            min_final_score_gap=("final_score_gap", "min"),
            max_final_score_gap=("final_score_gap", "max"),
        )
        .reset_index()
    )
    population_sensitivity_df["range_final_total_score"] = (
        population_sensitivity_df["max_final_total_score"]
        - population_sensitivity_df["min_final_total_score"]
    )
    population_sensitivity_df["range_final_score_gap"] = (
        population_sensitivity_df["max_final_score_gap"]
        - population_sensitivity_df["min_final_score_gap"]
    )
    population_sensitivity_df.to_csv(
        CSV_DIR / "population_sensitivity_by_build_institution.csv",
        index=False,
    )

    build_mode_sensitivity_df = (
        effect_df.groupby(["condition", "condition_label"], observed=False)
        .agg(
            average_crown_aware_effect_final_total_score=("delta_final_total_score", "mean"),
            min_crown_aware_effect_final_total_score=("delta_final_total_score", "min"),
            max_crown_aware_effect_final_total_score=("delta_final_total_score", "max"),
            average_crown_aware_effect_final_score_gap=("delta_final_score_gap", "mean"),
            min_crown_aware_effect_final_score_gap=("delta_final_score_gap", "min"),
            max_crown_aware_effect_final_score_gap=("delta_final_score_gap", "max"),
        )
        .reset_index()
    )
    build_mode_sensitivity_df["range_crown_aware_effect_final_total_score"] = (
        build_mode_sensitivity_df["max_crown_aware_effect_final_total_score"]
        - build_mode_sensitivity_df["min_crown_aware_effect_final_total_score"]
    )
    build_mode_sensitivity_df["range_crown_aware_effect_final_score_gap"] = (
        build_mode_sensitivity_df["max_crown_aware_effect_final_score_gap"]
        - build_mode_sensitivity_df["min_crown_aware_effect_final_score_gap"]
    )
    build_mode_sensitivity_df.to_csv(
        CSV_DIR / "build_mode_sensitivity_by_institution.csv",
        index=False,
    )

    return (
        mean_by_build_population_condition,
        effect_df,
        rankings_df,
        population_sensitivity_df,
        build_mode_sensitivity_df,
    )


def print_compact_grouped_means(mean_df: pd.DataFrame) -> None:
    """Print one compact score/gap/activity line for every factorial cell."""
    print("\nCompact grouped means by build mode, population, and condition:")
    ordered_df = mean_df.copy()
    ordered_df["build_mode_label"] = pd.Categorical(
        ordered_df["build_mode_label"],
        categories=BUILD_MODE_LABEL_ORDER,
        ordered=True,
    )
    ordered_df["population_label"] = pd.Categorical(
        ordered_df["population_label"],
        categories=POPULATION_LABEL_ORDER,
        ordered=True,
    )
    ordered_df["condition_label"] = pd.Categorical(
        ordered_df["condition_label"],
        categories=INSTITUTION_LABEL_ORDER,
        ordered=True,
    )
    ordered_df = ordered_df.sort_values(
        ["build_mode_label", "population_label", "condition_label"]
    )
    for _, row in ordered_df.iterrows():
        print(
            f"{row['build_mode_label']} | {row['population_label']} | {row['condition_label']}: "
            f"score={row['final_total_score']:.2f}, "
            f"gap={row['final_score_gap']:.2f}, "
            f"trades={row['cumulative_trades_executed']:.2f}, "
            f"idle_infra={row['final_idle_infrastructure']:.2f}"
        )


def print_diagnostics(
    mean_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    population_sensitivity_df: pd.DataFrame,
    build_mode_sensitivity_df: pd.DataFrame,
) -> None:
    """Print descriptive rankings, sensitivities, and build-mode contrasts."""
    print("\nDiagnostics:")

    print("\n1. Best institution by total score for each build mode × population:")
    best_score = rankings_df[rankings_df["rank_final_total_score"] == 1].copy()
    best_score["build_mode_label"] = pd.Categorical(
        best_score["build_mode_label"],
        categories=BUILD_MODE_LABEL_ORDER,
        ordered=True,
    )
    best_score["population_label"] = pd.Categorical(
        best_score["population_label"],
        categories=POPULATION_LABEL_ORDER,
        ordered=True,
    )
    for _, row in best_score.sort_values(["build_mode_label", "population_label"]).iterrows():
        print(f"- {row['build_mode_label']} | {row['population_label']}: {row['condition_label']}")

    print("\n2. Lowest-inequality institution for each build mode × population:")
    best_gap = rankings_df[rankings_df["rank_lowest_final_score_gap"] == 1].copy()
    best_gap["build_mode_label"] = pd.Categorical(
        best_gap["build_mode_label"],
        categories=BUILD_MODE_LABEL_ORDER,
        ordered=True,
    )
    best_gap["population_label"] = pd.Categorical(
        best_gap["population_label"],
        categories=POPULATION_LABEL_ORDER,
        ordered=True,
    )
    for _, row in best_gap.sort_values(["build_mode_label", "population_label"]).iterrows():
        print(f"- {row['build_mode_label']} | {row['population_label']}: {row['condition_label']}")

    print("\n3. Average crown-aware score effect by institution:")
    for _, row in build_mode_sensitivity_df.sort_values(
        "average_crown_aware_effect_final_total_score"
    ).iterrows():
        print(
            f"- {row['condition_label']}: "
            f"avg_delta_score={row['average_crown_aware_effect_final_total_score']:.2f}, "
            f"score_range={row['range_crown_aware_effect_final_total_score']:.2f}"
        )

    print("\n4. Average crown-aware score effect by population:")
    by_population = (
        effect_df.groupby(["population", "population_label"], observed=False)
        .agg(
            avg_delta_score=("delta_final_total_score", "mean"),
            avg_delta_gap=("delta_final_score_gap", "mean"),
        )
        .reset_index()
    )
    by_population["population_label"] = pd.Categorical(
        by_population["population_label"],
        categories=POPULATION_LABEL_ORDER,
        ordered=True,
    )
    for _, row in by_population.sort_values("population_label").iterrows():
        print(
            f"- {row['population_label']}: "
            f"avg_delta_score={row['avg_delta_score']:.2f}, "
            f"avg_delta_gap={row['avg_delta_gap']:.2f}"
        )

    print("\n5. Largest crown-aware score losses (population × institution):")
    loss_cols = [
        "population_label",
        "condition_label",
        "delta_final_total_score",
        "delta_final_idle_infrastructure",
        "delta_final_total_innovation",
    ]
    for _, row in effect_df.sort_values("delta_final_total_score").head(10)[loss_cols].iterrows():
        print(
            f"- {row['population_label']} | {row['condition_label']}: "
            f"delta_score={row['delta_final_total_score']:.2f}, "
            f"delta_idle={row['delta_final_idle_infrastructure']:.2f}, "
            f"delta_innovation={row['delta_final_total_innovation']:.2f}"
        )

    print("\n6. Most robust to crown-aware building:")
    effect_df["abs_delta_final_total_score"] = effect_df["delta_final_total_score"].abs()
    robust_cols = [
        "population_label",
        "condition_label",
        "delta_final_total_score",
        "delta_final_score_gap",
    ]
    for _, row in effect_df.sort_values("abs_delta_final_total_score").head(10)[robust_cols].iterrows():
        print(
            f"- {row['population_label']} | {row['condition_label']}: "
            f"delta_score={row['delta_final_total_score']:.2f}, "
            f"delta_gap={row['delta_final_score_gap']:.2f}"
        )

    print("\n7. Do decentralized institutions remain more behavior-sensitive than centralized ones?")
    decentralized = {
        "bilateral_trade",
        "bilateral_trade_3pass",
        "catch_up_bilateral_trade",
        "bottleneck_priority_bilateral_trade",
        "clearinghouse_bargaining",
    }
    centralized = {
        "public_pool",
        "subsidized_catch_up",
        "central_clearing",
        "equity_weighted_central",
        "central_clearing_capped",
    }
    for build_mode in BUILD_MODES:
        mode_df = population_sensitivity_df[population_sensitivity_df["build_mode"] == build_mode]
        decentralized_mean = mode_df[mode_df["condition"].isin(decentralized)]["range_final_total_score"].mean()
        centralized_mean = mode_df[mode_df["condition"].isin(centralized)]["range_final_total_score"].mean()
        print(
            f"- {BUILD_MODE_LABELS[build_mode]}: "
            f"decentralized mean score-range={decentralized_mean:.2f}, "
            f"centralized mean score-range={centralized_mean:.2f}"
        )

    print("\n8. Does crown-aware building increase idle infrastructure and innovation across most setups?")
    idle_positive = int((effect_df["delta_final_idle_infrastructure"] > 0).sum())
    innovation_positive = int((effect_df["delta_final_total_innovation"] > 0).sum())
    total_rows = len(effect_df)
    print(
        f"- idle infrastructure increases in {idle_positive}/{total_rows} population × institution setups"
    )
    print(
        f"- innovation increases in {innovation_positive}/{total_rows} population × institution setups"
    )

    print("\n9. Does crown-aware building change institution rankings?")
    dev_top = rankings_df[
        (rankings_df["build_mode"] == BUILD_MODE_DEVELOPMENT_ORIENTED)
        & (rankings_df["rank_final_total_score"] == 1)
    ][["population", "condition"]].rename(columns={"condition": "top_development"})
    crown_top = rankings_df[
        (rankings_df["build_mode"] == BUILD_MODE_CROWN_AWARE)
        & (rankings_df["rank_final_total_score"] == 1)
    ][["population", "condition"]].rename(columns={"condition": "top_crown"})
    ranking_change = dev_top.merge(crown_top, on="population")
    changed = ranking_change[ranking_change["top_development"] != ranking_change["top_crown"]]
    if changed.empty:
        print("- Top-scoring institution did not change for any population.")
    else:
        for _, row in changed.iterrows():
            print(
                f"- {POPULATION_LABELS[row['population']]}: "
                f"{INSTITUTION_LABELS[row['top_development']]} -> {INSTITUTION_LABELS[row['top_crown']]}"
            )


def main() -> None:
    """Run the full factorial, save tables and plots, and print diagnostics."""
    prepare_output_dirs()

    n_populations = len(POPULATION_ORDER)
    n_institutions = len(INSTITUTION_ORDER)
    total_games = len(BUILD_MODES) * N_SEEDS * n_populations * n_institutions

    print("Build mode × institution × utility population analysis:")
    print(f"BUILD_MODES: {BUILD_MODES}")
    print(f"N_SEEDS: {N_SEEDS}")
    print(f"ROUNDS: {ROUNDS}")
    print(f"number of populations: {n_populations}")
    print(f"number of institutions: {n_institutions}")
    print(f"total games planned: {total_games}")
    print(f"output folder: {OUTPUT_ROOT.resolve()}")

    summary_df, round_history_df, final_agents_df = run_all_experiments(
        seeds=range(N_SEEDS),
        rounds=ROUNDS,
        max_builds_per_agent_per_round=MAX_BUILDS_PER_AGENT_PER_ROUND,
    )
    (
        mean_df,
        effect_df,
        rankings_df,
        population_sensitivity_df,
        build_mode_sensitivity_df,
    ) = save_summary_csvs(
        summary_df=summary_df,
        round_history_df=round_history_df,
        final_agents_df=final_agents_df,
    )
    # Generate figures and report tables from the newly saved experiment CSVs.
    from experiments.analysis.analyze_utility_results import (
        regenerate_utility_outputs,
    )

    regenerate_utility_outputs()

    # Print compact cell means as a quick check of the completed run.
    print_compact_grouped_means(mean_df)


if __name__ == "__main__":
    main()
