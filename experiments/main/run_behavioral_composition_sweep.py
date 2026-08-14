"""Run the behavioral-composition sweeps used in the paper."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from engine.build_rules import BuildRules
from engine.game import Game
from engine.institutions import (
    BilateralTradeInstitution,
    CentralMarketClearingInstitution,
    EquityWeightedCentralClearingInstitution,
    NoTradeInstitution,
    PublicPoolInstitution,
)
from engine.policies import (
    BUILD_MODE_CHOICES,
    BUILD_MODE_DEVELOPMENT_ORIENTED,
    CooperativeTradePolicy,
    CompetitiveTradePolicy,
    HoardingTradePolicy,
    NeedBasedTradePolicy,
)
from engine.resources import create_random_balanced_access_profiles
from experiments.main.run_institution_behavior_analysis import (
    MAX_BUILDS_PER_AGENT_PER_ROUND,
    RNG_STREAM_ACCESS,
    RNG_STREAM_INSTITUTION,
    RNG_STREAM_MIXED_ASSIGNMENT,
    RNG_STREAM_PRODUCTION,
    create_agents,
    make_rng,
)


OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs" / "composition_sweep"
ROUNDS = 20
AGENT_COUNT = 5

SWEEPS: dict[str, dict[str, Any]] = {
    "cooperative_to_hoarding": {
        "label": "Cooperative \u2192 hoarding",
        "restrictive_label": "Hoarding agents",
        "baseline_policy": CooperativeTradePolicy,
        "restrictive_policy": HoardingTradePolicy,
    },
    "need_based_to_competitive": {
        "label": "Need-based \u2192 competitive",
        "restrictive_label": "Competitive agents",
        "baseline_policy": NeedBasedTradePolicy,
        "restrictive_policy": CompetitiveTradePolicy,
    },
}

CONDITIONS: dict[str, dict[str, Any]] = {
    "no_trade": {
        "label": "No trade",
        "factory": NoTradeInstitution,
    },
    "bilateral_3pass": {
        "label": "Bilateral 3-pass",
        "factory": lambda: BilateralTradeInstitution(max_bargaining_passes=3),
    },
    "public_pool": {
        "label": "Public pool",
        "factory": lambda: PublicPoolInstitution(
            max_allocations_per_round=3,
            prioritize_low_score=True,
        ),
    },
    "central_cap2": {
        "label": "Central capped (2)",
        "factory": lambda: CentralMarketClearingInstitution(max_trades_per_round=2),
    },
    "central_full": {
        "label": "Central clearing",
        "factory": lambda: CentralMarketClearingInstitution(max_trades_per_round=None),
    },
    "equity_central": {
        "label": "Equity central",
        "factory": lambda: EquityWeightedCentralClearingInstitution(
            equity_weight=1.0,
            max_trades_per_round=None,
        ),
    },
}

SUMMARY_COLUMNS = [
    "seed",
    "sweep",
    "sweep_label",
    "restrictive_label",
    "restrictive_count",
    "restrictive_share",
    "condition",
    "condition_label",
    "final_total_score",
    "final_min_score",
    "final_score_gap",
    "final_idle_infrastructure",
    "cumulative_trades_executed",
]
FINAL_AGENT_COLUMNS = [
    "seed",
    "sweep",
    "sweep_label",
    "restrictive_count",
    "condition",
    "condition_label",
    "agent_id",
    "policy_class_name",
    "score",
    "infrastructure",
    "production_sites",
    "advanced_sites",
    "innovation",
]
ROUND_COLUMNS = [
    "seed",
    "sweep",
    "sweep_label",
    "restrictive_count",
    "restrictive_share",
    "condition",
    "condition_label",
    "round",
    "institution",
    "total_score",
    "min_score",
    "max_score",
    "score_gap",
    "infrastructure_leader_id",
    "innovation_leader_id",
    "infrastructure_leader_bonus_active",
    "innovation_leader_bonus_active",
    "total_resources",
    "builds_applied",
    "trades_proposed",
    "trades_executed",
    "total_infrastructure",
    "idle_infrastructure",
    "total_production_sites",
    "total_advanced_sites",
    "total_innovation",
    "total_materials",
    "total_components",
    "total_food",
    "total_energy",
    "total_knowledge",
]


def parse_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def parse_names(raw: str, allowed: dict[str, Any], label: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}")
    return names


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def policies_for(seed: int, sweep_name: str, restrictive_count: int, build_mode: str):
    spec = SWEEPS[sweep_name]
    order = list(range(AGENT_COUNT))
    make_rng(seed, RNG_STREAM_MIXED_ASSIGNMENT).shuffle(order)
    restrictive_ids = set(order[:restrictive_count])

    return {
        agent_id: (
            spec["restrictive_policy"]
            if agent_id in restrictive_ids
            else spec["baseline_policy"]
        )(build_mode=build_mode)
        for agent_id in range(AGENT_COUNT)
    }


def run_game(
    *,
    seed: int,
    sweep_name: str,
    restrictive_count: int,
    condition_name: str,
    build_mode: str,
):
    profiles = list(create_random_balanced_access_profiles(make_rng(seed, RNG_STREAM_ACCESS)).values())
    agents = create_agents(profiles)
    game = Game(
        agents=agents,
        policies=policies_for(seed, sweep_name, restrictive_count, build_mode),
        build_rules=BuildRules(),
        institution=CONDITIONS[condition_name]["factory"](),
        rng=make_rng(seed, RNG_STREAM_INSTITUTION),
        production_rng=make_rng(seed, RNG_STREAM_PRODUCTION),
        max_builds_per_agent_per_round=MAX_BUILDS_PER_AGENT_PER_ROUND,
    )

    for _ in range(ROUNDS):
        game.step()

    return game


def collect_rows(seeds: list[int], counts: list[int], sweeps: list[str], conditions: list[str], build_mode: str):
    summary_rows = []
    final_agent_rows = []
    round_rows = []

    for sweep_name in sweeps:
        sweep_spec = SWEEPS[sweep_name]
        for restrictive_count in counts:
            restrictive_share = restrictive_count / AGENT_COUNT
            for condition_name in conditions:
                condition_spec = CONDITIONS[condition_name]
                for seed in seeds:
                    game = run_game(
                        seed=seed,
                        sweep_name=sweep_name,
                        restrictive_count=restrictive_count,
                        condition_name=condition_name,
                        build_mode=build_mode,
                    )
                    final_metrics = game.history[-1]
                    base = {
                        "seed": seed,
                        "sweep": sweep_name,
                        "sweep_label": sweep_spec["label"],
                        "restrictive_count": restrictive_count,
                        "restrictive_share": restrictive_share,
                        "condition": condition_name,
                        "condition_label": condition_spec["label"],
                    }
                    summary_rows.append(
                        {
                            **base,
                            "restrictive_label": sweep_spec["restrictive_label"],
                            "final_total_score": final_metrics["total_score"],
                            "final_min_score": final_metrics["min_score"],
                            "final_score_gap": final_metrics["score_gap"],
                            "final_idle_infrastructure": final_metrics["idle_infrastructure"],
                            "cumulative_trades_executed": sum(
                                row["trades_executed"] for row in game.history
                            ),
                        }
                    )

                    for agent in game.agents:
                        final_agent_rows.append(
                            {
                                "seed": seed,
                                "sweep": sweep_name,
                                "sweep_label": sweep_spec["label"],
                                "restrictive_count": restrictive_count,
                                "condition": condition_name,
                                "condition_label": condition_spec["label"],
                                "agent_id": agent.id,
                                "policy_class_name": game.policies[agent.id].__class__.__name__,
                                "score": agent.score,
                                "infrastructure": agent.infrastructure,
                                "production_sites": agent.production_sites,
                                "advanced_sites": agent.advanced_sites,
                                "innovation": agent.innovation,
                            }
                        )

                    for metrics in game.history:
                        round_rows.append({**base, **metrics})

    return (
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
        pd.DataFrame(final_agent_rows, columns=FINAL_AGENT_COLUMNS),
        pd.DataFrame(round_rows, columns=ROUND_COLUMNS),
    )


def save_tables(summary_df: pd.DataFrame, output_root: Path) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    mean_df = (
        summary_df.groupby(
            [
                "sweep",
                "sweep_label",
                "restrictive_label",
                "restrictive_count",
                "restrictive_share",
                "condition",
                "condition_label",
            ],
            as_index=False,
        )[
            [
                "final_total_score",
                "final_min_score",
                "final_score_gap",
                "cumulative_trades_executed",
            ]
        ]
        .mean()
    )
    mean_df.to_csv(tables_dir / "mean_by_composition_institution.csv", index=False)

    pivot = mean_df.pivot_table(
        index=["sweep", "restrictive_count"],
        columns="condition",
        values="final_total_score",
    ).reset_index()
    pool_central = pd.DataFrame(
        {
            "restrictive_count": pivot["restrictive_count"],
            "pool": pivot["public_pool"],
            "central": pivot["central_full"],
            "difference": pivot["public_pool"] - pivot["central_full"],
            "sweep": pivot["sweep"],
        }
    )
    pool_central.to_csv(
        tables_dir / "public_pool_minus_central_by_composition.csv",
        index=False,
    )


def write_metadata(output_root: Path, args: argparse.Namespace, rows: dict[str, int]) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "build_mode": args.build_mode,
        "rounds": ROUNDS,
        "agent_count": AGENT_COUNT,
        "max_builds_per_agent_per_round": MAX_BUILDS_PER_AGENT_PER_ROUND,
        "seeds": args.seeds,
        "restrictive_counts": args.restrictive_counts,
        "sweeps": args.sweeps,
        "conditions": args.conditions,
        "rng_streams": {
            "access": RNG_STREAM_ACCESS,
            "production": RNG_STREAM_PRODUCTION,
            "institution": RNG_STREAM_INSTITUTION,
            "mixed_assignment": RNG_STREAM_MIXED_ASSIGNMENT,
        },
        "rows": rows,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--build-mode", choices=sorted(BUILD_MODE_CHOICES), default=BUILD_MODE_DEVELOPMENT_ORIENTED)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(100)))
    parser.add_argument("--restrictive-counts", default="0,1,2,3,4,5")
    parser.add_argument("--sweeps", default=",".join(SWEEPS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = "0,1,2"
        args.restrictive_counts = "0,3,5"

    args.seeds = parse_ints(args.seeds)
    args.restrictive_counts = parse_ints(args.restrictive_counts)
    args.sweeps = parse_names(args.sweeps, SWEEPS, "sweeps")
    args.conditions = parse_names(args.conditions, CONDITIONS, "conditions")

    output_root = args.output_root.resolve()
    csv_dir = output_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary_df, final_agents_df, round_history_df = collect_rows(
        args.seeds,
        args.restrictive_counts,
        args.sweeps,
        args.conditions,
        args.build_mode,
    )

    summary_df.to_csv(csv_dir / "summary_by_seed.csv", index=False)
    final_agents_df.to_csv(csv_dir / "final_agents.csv", index=False)
    round_history_df.to_csv(csv_dir / "round_history.csv", index=False)
    save_tables(summary_df, output_root)
    write_metadata(
        output_root,
        args,
        {
            "summary_by_seed": len(summary_df),
            "final_agents": len(final_agents_df),
            "round_history": len(round_history_df),
        },
    )

    print(f"Wrote composition-sweep outputs to {output_root}")


if __name__ == "__main__":
    main()
