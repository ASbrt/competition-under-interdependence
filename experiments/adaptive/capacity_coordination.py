"""Online institutional-selection environment with regenerating coordination capacity.

This module replaces the former fixed token budget. A meta-planner selects one
exchange institution after production in every round. Coordinated institutions
consume a real shared capacity stock, which then recovers gradually. Capacity
costs therefore change the future feasible action set rather than existing only
as a penalty inside the learner's reward.

The environment intentionally has no Gym dependency. It exposes a small API
suitable for online tabular Q-learning and later function approximation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Callable, Mapping, Sequence

import numpy as np

from engine.agents import Agent
from engine.build_rules import BuildRules
from engine.game import Game
from engine.institutions import (
    BilateralTradeInstitution,
    CentralMarketClearingInstitution,
    ClearinghouseBargainingInstitution,
    EquityWeightedCentralClearingInstitution,
    Institution,
    InstitutionResult,
    RoundLocalPublicPoolInstitution,
    SubsidizedCatchUpInstitution,
)
from engine.policies import AgentPolicy
from engine.resources import ResourceAccessProfile


PolicyBuilder = Callable[[Sequence[Agent]], dict[int, AgentPolicy]]
PolicySchedule = Mapping[int, PolicyBuilder]


def _bundle_units(bundle) -> int:
    if not isinstance(bundle, dict):
        return 0
    return sum(max(0, int(quantity)) for quantity in bundle.values())


def institution_workload_units(result: InstitutionResult) -> int:
    """Return a comparable count of resource-handling operations.

    Reciprocal exchanges count all units moved in both directions. Subsidies
    count transferred units. Public-pool operation counts contributions,
    allocations, and returned leftovers because each requires administration.
    The generic event counters are intentionally not used because their meaning
    differs across institutions.
    """
    details = result.details or {}

    if "executed_offers" in details:
        return sum(
            _bundle_units(row.get("offered_bundle", {}))
            + _bundle_units(row.get("requested_bundle", {}))
            for row in details.get("executed_offers", [])
        )

    if "subsidies_executed" in details:
        return sum(
            max(0, int(row.get("quantity", 0)))
            for row in details.get("subsidies_executed", [])
        )

    if "pool_contributions" in details or "pool_allocations" in details:
        contribution_units = sum(
            _bundle_units(row.get("bundle", {}))
            for row in details.get("pool_contributions", [])
        )
        allocation_units = sum(
            _bundle_units(row.get("bundle", {}))
            for row in details.get("pool_allocations", [])
        )
        return_units = sum(
            _bundle_units(row.get("bundle", {}))
            for row in details.get("pool_returns", [])
        )
        return contribution_units + allocation_units + return_units

    return 0


@dataclass(frozen=True)
class CapacityActionSpec:
    """One institutional action and its governance-capacity requirements."""

    name: str
    label: str
    factory: Callable[[], Institution]
    base_capacity_cost: int
    workload_units_per_cost: int
    max_workload_cost: int
    switch_capacity_cost: int = 1

    def __post_init__(self) -> None:
        if self.base_capacity_cost < 0:
            raise ValueError("base_capacity_cost must be non-negative.")
        if self.workload_units_per_cost < 0:
            raise ValueError("workload_units_per_cost must be non-negative.")
        if self.max_workload_cost < 0:
            raise ValueError("max_workload_cost must be non-negative.")
        if self.switch_capacity_cost < 0:
            raise ValueError("switch_capacity_cost must be non-negative.")
        if self.max_workload_cost > 0 and self.workload_units_per_cost <= 0:
            raise ValueError(
                "Positive max_workload_cost requires workload_units_per_cost > 0."
            )

    def switch_cost(self, previous_action: str | None) -> int:
        """Charge reconfiguration only when entering a coordinated regime.

        Bilateral bargaining is the default decentralized fallback. Returning
        to it is free, while entering a different non-bilateral institution
        requires setup capacity.
        """
        if self.name == "bilateral_3pass":
            return 0
        if previous_action in (None, self.name):
            return 0
        return self.switch_capacity_cost

    def commitment_cost(self, previous_action: str | None) -> int:
        """Capacity that must be available before committing to the action.

        The planner reserves the maximum bounded workload component. Realized
        cost may be lower when the institution performs little work.
        """
        return (
            self.base_capacity_cost
            + self.switch_cost(previous_action)
            + self.max_workload_cost
        )

    def realized_cost_components(
        self,
        *,
        previous_action: str | None,
        workload_units: int,
    ) -> tuple[int, int, int, int]:
        switch = self.switch_cost(previous_action)
        workload = 0
        if workload_units > 0 and self.max_workload_cost > 0:
            workload = min(
                self.max_workload_cost,
                ceil(workload_units / self.workload_units_per_cost),
            )
        total = self.base_capacity_cost + switch + workload
        return self.base_capacity_cost, switch, workload, total


DEFAULT_ACTION_SPECS: tuple[CapacityActionSpec, ...] = (
    CapacityActionSpec(
        name="bilateral_3pass",
        label="Bilateral 3-pass",
        factory=lambda: BilateralTradeInstitution(max_bargaining_passes=3),
        base_capacity_cost=0,
        workload_units_per_cost=0,
        max_workload_cost=0,
        switch_capacity_cost=0,
    ),
    CapacityActionSpec(
        name="clearinghouse",
        label="Clearinghouse",
        factory=lambda: ClearinghouseBargainingInstitution(max_bargaining_passes=1),
        base_capacity_cost=1,
        workload_units_per_cost=6,
        max_workload_cost=1,
    ),
    CapacityActionSpec(
        name="public_pool",
        label="Round-local public pool",
        factory=lambda: RoundLocalPublicPoolInstitution(
            max_allocations_per_round=3,
            prioritize_low_score=True,
        ),
        base_capacity_cost=1,
        workload_units_per_cost=6,
        max_workload_cost=2,
    ),
    CapacityActionSpec(
        name="subsidized_catch_up",
        label="Subsidized catch-up",
        factory=lambda: SubsidizedCatchUpInstitution(max_subsidies_per_round=2),
        base_capacity_cost=1,
        workload_units_per_cost=2,
        max_workload_cost=1,
    ),
    CapacityActionSpec(
        name="central_cap2",
        label="Central capped (2)",
        factory=lambda: CentralMarketClearingInstitution(max_trades_per_round=2),
        base_capacity_cost=2,
        workload_units_per_cost=4,
        max_workload_cost=1,
    ),
    CapacityActionSpec(
        name="equity_cap2",
        label="Equity central capped (2)",
        factory=lambda: EquityWeightedCentralClearingInstitution(
            equity_weight=1.0,
            max_trades_per_round=2,
        ),
        base_capacity_cost=2,
        workload_units_per_cost=4,
        max_workload_cost=1,
    ),
    CapacityActionSpec(
        name="central_full",
        label="Central clearing",
        factory=lambda: CentralMarketClearingInstitution(max_trades_per_round=None),
        base_capacity_cost=3,
        workload_units_per_cost=6,
        max_workload_cost=2,
    ),
)


@dataclass(frozen=True)
class CapacityObservation:
    """Post-production economic state observed by the institutional planner."""

    round_number: int
    total_rounds: int
    rounds_remaining: int
    coordination_capacity: int
    max_coordination_capacity: int
    capacity_recovery_per_round: int
    previous_action: str
    total_score: int
    mean_score: float
    min_score: int
    max_score: int
    score_gap: int
    bottom_two_mean_score: float
    total_resources: int
    total_missing_units: int
    agents_one_unit_short: int
    blocked_agents: int
    current_requests: int
    idle_infrastructure: int
    total_production_sites: int
    total_advanced_sites: int
    voluntary_matchable_requests: int
    voluntary_acceptable_offers: int
    feasible_voluntary_trade_units: int
    voluntary_match_rate: float
    recent_voluntary_match_rate: float
    voluntary_match_rate_drop: float
    shortage_concentration: float

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class CapacityInstitutionEnvironment:
    """Round-level environment with regenerating governance capacity."""

    def __init__(
        self,
        *,
        profiles: Sequence[ResourceAccessProfile],
        initial_policy_builder: PolicyBuilder,
        seed: int,
        total_rounds: int = 20,
        max_coordination_capacity: int = 8,
        initial_coordination_capacity: int | None = None,
        capacity_recovery_per_round: int = 1,
        max_builds_per_agent_per_round: int = 4,
        equity_weight: float = 0.25,
        action_specs: Sequence[CapacityActionSpec] = DEFAULT_ACTION_SPECS,
        policy_schedule: PolicySchedule | None = None,
    ):
        if total_rounds <= 0:
            raise ValueError("total_rounds must be positive.")
        if max_coordination_capacity <= 0:
            raise ValueError("max_coordination_capacity must be positive.")
        if capacity_recovery_per_round < 0:
            raise ValueError("capacity_recovery_per_round must be non-negative.")
        if not 0 <= equity_weight <= 1:
            raise ValueError("equity_weight must lie in [0, 1].")
        if max_builds_per_agent_per_round <= 0:
            raise ValueError("max_builds_per_agent_per_round must be positive.")

        initial_capacity = (
            max_coordination_capacity
            if initial_coordination_capacity is None
            else initial_coordination_capacity
        )
        if not 0 <= initial_capacity <= max_coordination_capacity:
            raise ValueError(
                "initial_coordination_capacity must lie between 0 and the maximum."
            )

        self.seed = int(seed)
        self.total_rounds = int(total_rounds)
        self.max_coordination_capacity = int(max_coordination_capacity)
        self.coordination_capacity = int(initial_capacity)
        self.capacity_recovery_per_round = int(capacity_recovery_per_round)
        self.equity_weight = float(equity_weight)
        self.policy_schedule = dict(policy_schedule or {})

        self._action_specs = {spec.name: spec for spec in action_specs}
        if "bilateral_3pass" not in self._action_specs:
            raise ValueError("A bilateral_3pass fallback action is required.")
        if self._action_specs["bilateral_3pass"].commitment_cost(None) != 0:
            raise ValueError("The bilateral fallback must require zero capacity.")
        self._institutions = {
            name: spec.factory()
            for name, spec in self._action_specs.items()
        }
        self.previous_action = "bilateral_3pass"

        agents = [
            Agent(id=index, access_profile=profile)
            for index, profile in enumerate(profiles)
        ]
        policies = initial_policy_builder(agents)
        expected_ids = {agent.id for agent in agents}
        if set(policies) != expected_ids:
            raise ValueError("Policy builder must return one policy per agent id.")

        institution_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 2])
        )
        production_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 1])
        )
        self.game = Game(
            agents=agents,
            policies=policies,
            build_rules=BuildRules(),
            institution=self._institutions["bilateral_3pass"],
            rng=institution_rng,
            production_rng=production_rng,
            max_builds_per_agent_per_round=max_builds_per_agent_per_round,
        )

        self._pending_produced: dict[int, dict[str, int]] | None = None
        self._decision_state = None
        self._terminated = False
        self._cached_observation: CapacityObservation | None = None
        self.action_history: list[dict[str, object]] = []
        self._begin_round()

    @property
    def action_specs(self) -> Mapping[str, CapacityActionSpec]:
        return self._action_specs

    @property
    def terminated(self) -> bool:
        return self._terminated

    def _apply_scheduled_policy_change(self) -> None:
        """Apply exogenous behavior transitions used by evaluation scenarios."""
        builder = self.policy_schedule.get(self.game.round_number)
        if builder is None:
            return
        policies = builder(self.game.agents)
        expected_ids = {agent.id for agent in self.game.agents}
        if set(policies) != expected_ids:
            raise ValueError(
                f"Policy schedule at round {self.game.round_number} returned "
                "an invalid agent-id set."
            )
        self.game.policies = policies

    def _begin_round(self) -> None:
        if self.game.round_number >= self.total_rounds:
            self._terminated = True
            self._pending_produced = None
            self._decision_state = None
            self._cached_observation = None
            return
        self._cached_observation = None
        self._apply_scheduled_policy_change()
        self._pending_produced = self.game.produce_resources()
        self._decision_state = self.game.get_state()

    def available_actions(self) -> tuple[str, ...]:
        """Return actions whose bounded commitment fits current capacity."""
        return tuple(
            name
            for name, spec in self._action_specs.items()
            if name == "bilateral_3pass"
            or spec.commitment_cost(self.previous_action)
            <= self.coordination_capacity
        )

    def action_commitment_costs(self) -> dict[str, int]:
        return {
            name: spec.commitment_cost(self.previous_action)
            for name, spec in self._action_specs.items()
        }

    def _voluntary_exchange_diagnostic(
        self,
        state,
    ) -> tuple[int, int, int, int, float]:
        requests = []
        for agent_state in state.agents:
            policy = self.game.policies[agent_state.id]
            requests.extend(
                policy.create_trade_requests(
                    agent_id=agent_state.id,
                    state=state,
                    build_rules=self.game.build_rules,
                )[:1]
            )

        matchable_requests = 0
        acceptable_offers = 0
        feasible_trade_units = 0
        for request in requests:
            request_has_match = False
            for responder_state in state.agents:
                if responder_state.id == request.requester_id:
                    continue
                responder_policy = self.game.policies[responder_state.id]
                offer = responder_policy.respond_to_trade_request(
                    agent_id=responder_state.id,
                    request=request,
                    state=state,
                    build_rules=self.game.build_rules,
                )
                if offer is None:
                    continue
                if not self.game.can_pay_bundle(
                    offer.responder_id, offer.offered_bundle
                ):
                    continue
                if not self.game.can_pay_bundle(
                    offer.requester_id, offer.requested_bundle
                ):
                    continue
                requester_policy = self.game.policies[request.requester_id]
                if not requester_policy.accepts_trade_offer(
                    agent_id=request.requester_id,
                    offer=offer,
                    state=state,
                    build_rules=self.game.build_rules,
                ):
                    continue
                acceptable_offers += 1
                feasible_trade_units += _bundle_units(offer.offered_bundle)
                feasible_trade_units += _bundle_units(offer.requested_bundle)
                request_has_match = True
            if request_has_match:
                matchable_requests += 1

        match_rate = matchable_requests / len(requests) if requests else 1.0
        return (
            len(requests),
            matchable_requests,
            acceptable_offers,
            feasible_trade_units,
            float(match_rate),
        )

    def observation(self) -> CapacityObservation:
        if self._cached_observation is not None:
            return self._cached_observation
        if self._terminated or self._decision_state is None:
            raise RuntimeError("No decision observation exists after termination.")

        state = self._decision_state
        scores = [agent_state.score for agent_state in state.agents]
        sorted_scores = sorted(scores)
        total_missing_units = 0
        missing_by_resource: dict[str, int] = {}
        agents_one_unit_short = 0
        blocked_agents = 0
        for agent_state in state.agents:
            policy = self.game.policies[agent_state.id]
            target = policy.choose_build_target(
                agent_id=agent_state.id,
                state=state,
                build_rules=self.game.build_rules,
            )
            missing = self.game.build_rules.missing_resources_for_build(
                agent_state,
                target.build_name,
            )
            missing_units = sum(missing.values())
            total_missing_units += missing_units
            for resource, quantity in missing.items():
                missing_by_resource[resource] = (
                    missing_by_resource.get(resource, 0) + int(quantity)
                )
            agents_one_unit_short += int(missing_units == 1)
            blocked_agents += int(missing_units > 0)

        (
            current_requests,
            voluntary_matchable_requests,
            voluntary_acceptable_offers,
            feasible_voluntary_trade_units,
            voluntary_match_rate,
        ) = self._voluntary_exchange_diagnostic(state)

        previous_rates = [
            float(row["voluntary_match_rate"])
            for row in self.action_history[-3:]
            if "voluntary_match_rate" in row
        ]
        recent_rates = (previous_rates + [voluntary_match_rate])[-3:]
        recent_voluntary_match_rate = float(np.mean(recent_rates))
        voluntary_match_rate_drop = (
            max(0.0, float(np.mean(previous_rates)) - voluntary_match_rate)
            if previous_rates
            else 0.0
        )
        shortage_concentration = (
            max(missing_by_resource.values()) / total_missing_units
            if total_missing_units > 0
            else 0.0
        )

        idle_infrastructure = sum(
            max(
                0,
                agent_state.infrastructure
                - 2 * (
                    agent_state.production_sites
                    + agent_state.advanced_sites
                ),
            )
            for agent_state in state.agents
        )

        self._cached_observation = CapacityObservation(
            round_number=self.game.round_number,
            total_rounds=self.total_rounds,
            rounds_remaining=self.total_rounds - self.game.round_number,
            coordination_capacity=self.coordination_capacity,
            max_coordination_capacity=self.max_coordination_capacity,
            capacity_recovery_per_round=self.capacity_recovery_per_round,
            previous_action=self.previous_action,
            total_score=sum(scores),
            mean_score=float(np.mean(scores)),
            min_score=min(scores),
            max_score=max(scores),
            score_gap=max(scores) - min(scores),
            bottom_two_mean_score=float(np.mean(sorted_scores[:2])),
            total_resources=sum(
                sum(agent_state.stock.values()) for agent_state in state.agents
            ),
            total_missing_units=total_missing_units,
            agents_one_unit_short=agents_one_unit_short,
            blocked_agents=blocked_agents,
            current_requests=current_requests,
            idle_infrastructure=idle_infrastructure,
            total_production_sites=sum(
                agent_state.production_sites for agent_state in state.agents
            ),
            total_advanced_sites=sum(
                agent_state.advanced_sites for agent_state in state.agents
            ),
            voluntary_matchable_requests=voluntary_matchable_requests,
            voluntary_acceptable_offers=voluntary_acceptable_offers,
            feasible_voluntary_trade_units=feasible_voluntary_trade_units,
            voluntary_match_rate=voluntary_match_rate,
            recent_voluntary_match_rate=recent_voluntary_match_rate,
            voluntary_match_rate_drop=voluntary_match_rate_drop,
            shortage_concentration=float(shortage_concentration),
        )
        return self._cached_observation

    def welfare(self, *, mean_score: float, min_score: float) -> float:
        """Inclusive-development welfare optimized by the planner."""
        return (
            (1.0 - self.equity_weight) * float(mean_score)
            + self.equity_weight * float(min_score)
        )

    def step(
        self,
        action_name: str,
    ) -> tuple[CapacityObservation | None, float, bool, dict[str, object]]:
        if self._terminated:
            raise RuntimeError("Cannot step a terminated environment.")
        if action_name not in self._action_specs:
            raise KeyError(f"Unknown action {action_name!r}.")
        if action_name not in self.available_actions():
            raise ValueError(
                f"Action {action_name!r} is unavailable at capacity "
                f"{self.coordination_capacity}."
            )
        if self._decision_state is None or self._pending_produced is None:
            raise RuntimeError("Environment is not at a decision point.")

        spec = self._action_specs[action_name]
        pre = self.observation()
        welfare_before = self.welfare(
            mean_score=pre.mean_score,
            min_score=pre.min_score,
        )
        capacity_before = self.coordination_capacity
        previous_action = self.previous_action

        self.game.institution = self._institutions[action_name]
        institution_result = self.game.institution.resolve(
            game=self.game,
            state=self._decision_state,
        )
        builds_applied = self.game.apply_policy_builds()
        metrics = self.game.compute_metrics(
            produced=self._pending_produced,
            institution_result=institution_result,
            builds_applied=builds_applied,
        )

        workload_units = institution_workload_units(institution_result)
        (
            base_cost,
            switch_cost,
            workload_cost,
            realized_cost,
        ) = spec.realized_cost_components(
            previous_action=previous_action,
            workload_units=workload_units,
        )
        if realized_cost > capacity_before:
            raise RuntimeError(
                "Realized capacity cost exceeded the reserved commitment."
            )
        capacity_after = min(
            self.max_coordination_capacity,
            capacity_before
            - realized_cost
            + self.capacity_recovery_per_round,
        )
        self.coordination_capacity = int(capacity_after)
        self.previous_action = action_name

        welfare_after = self.welfare(
            mean_score=float(metrics["total_score"]) / len(self.game.agents),
            min_score=float(metrics["min_score"]),
        )
        reward = welfare_after - welfare_before

        metrics.update(
            {
                "planner_action": action_name,
                "planner_action_label": spec.label,
                "coordination_capacity_before": capacity_before,
                "coordination_capacity_after": self.coordination_capacity,
                "capacity_commitment": spec.commitment_cost(previous_action),
                "capacity_base_cost": base_cost,
                "capacity_switch_cost": switch_cost,
                "capacity_workload_cost": workload_cost,
                "capacity_realized_cost": realized_cost,
                "institution_workload_units": workload_units,
                "planner_welfare_before": welfare_before,
                "planner_welfare_after": welfare_after,
                "planner_reward": reward,
                **{
                    f"observation_{key}": value
                    for key, value in pre.as_dict().items()
                },
            }
        )
        self.game.history.append(metrics)

        self.action_history.append(
            {
                "round": pre.round_number,
                "action": action_name,
                "action_label": spec.label,
                "capacity_before": capacity_before,
                "capacity_after": self.coordination_capacity,
                "capacity_realized_cost": realized_cost,
                "workload_units": workload_units,
                "reward": reward,
                **pre.as_dict(),
            }
        )

        self.game.round_number += 1
        self._pending_produced = None
        self._decision_state = None
        self._begin_round()

        next_observation = None if self._terminated else self.observation()
        info = {
            "metrics": metrics,
            "action_name": action_name,
            "action_label": spec.label,
            "capacity_before": capacity_before,
            "capacity_after": self.coordination_capacity,
            "realized_capacity_cost": realized_cost,
            "workload_units": workload_units,
            "welfare_before": welfare_before,
            "welfare_after": welfare_after,
        }
        return next_observation, float(reward), self._terminated, info
