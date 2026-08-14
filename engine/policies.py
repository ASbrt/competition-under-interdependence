"""
Utility-based agent decision rules.

Policies inspect `GameState` plus `BuildRules` and decide what an agent wants
to build or trade. They do not directly mutate agents; instead they return
small action/request objects that `Game` or an `Institution` can later execute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import random

from engine.actions import (
    BuildAction,
    BuildDecision,
    BuildTarget,
    NoBuildAction,
    ResourceBundle,
    TradeOffer,
    TradeRequest,
)
from engine.build_rules import BuildName, BuildRules
from engine.resources import RESOURCES, ResourceName
from engine.state import GameState

BUILD_MODE_DEVELOPMENT_ORIENTED = "development_oriented"
BUILD_MODE_CROWN_AWARE = "crown_aware"
BUILD_MODE_CHOICES = {
    BUILD_MODE_DEVELOPMENT_ORIENTED,
    BUILD_MODE_CROWN_AWARE,
}


@dataclass(frozen=True)
class UtilityWeights:
    """Compact, interpretable preference profile for one rule-based agent."""

    own_score: float = 1.0
    social_welfare: float = 0.0
    relative_advantage: float = 0.0
    equity: float = 0.0
    productive_capacity: float = 1.0
    site_progress: float = 1.0
    innovation_preference: float = 0.0
    security: float = 0.2
    progress: float = 1.0
    crown: float = 0.0


class AgentPolicy(ABC):
    """Base class for all agent decision rules."""

    @abstractmethod
    def choose_build_target(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> BuildTarget:
        pass

    @abstractmethod
    def choose_build_action(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> BuildDecision:
        pass

    @abstractmethod
    def create_trade_requests(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> list[TradeRequest]:
        pass

    @abstractmethod
    def respond_to_trade_request(
        self,
        agent_id: int,
        request: TradeRequest,
        state: GameState,
        build_rules: BuildRules,
    ) -> TradeOffer | None:
        pass

    @abstractmethod
    def accepts_trade_offer(
        self,
        agent_id: int,
        offer: TradeOffer,
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        pass

    def choose_pool_contribution(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> dict[str, int] | None:
        """Return None for institutional fallback, or an explicit contribution."""
        return None

    def accepts_pool_allocation(
        self,
        agent_id: int,
        bundle: dict[str, int],
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        return True

    def trade_priority_weight(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> float:
        return 1.0


class RandomBuildPolicy(AgentPolicy):
    """Simple debugging policy."""

    def __init__(self, rng):
        self.rng = rng

    def choose_build_target(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> BuildTarget:
        agent_state = state.agent_state(agent_id)
        candidates = [
            build_name
            for build_name in build_rules.projects
            if build_rules.has_structural_prerequisites(agent_state, build_name)
        ]
        if candidates:
            return BuildTarget(agent_id=agent_id, build_name=str(self.rng.choice(candidates)))
        return BuildTarget(agent_id=agent_id, build_name="infrastructure")

    def choose_build_action(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> BuildDecision:
        available = build_rules.available_builds(state.agent_state(agent_id))
        if available:
            return BuildAction(agent_id=agent_id, build_name=str(self.rng.choice(available)))
        return NoBuildAction(agent_id=agent_id)

    def create_trade_requests(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> list[TradeRequest]:
        return []

    def respond_to_trade_request(
        self,
        agent_id: int,
        request: TradeRequest,
        state: GameState,
        build_rules: BuildRules,
    ) -> TradeOffer | None:
        return None

    def accepts_trade_offer(
        self,
        agent_id: int,
        offer: TradeOffer,
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        return False


class UtilityBasedTradePolicy(AgentPolicy):
    """
    Shared utility-driven policy.

    Agents differ by stable weights rather than by disjoint hard-coded offer
    rules. The same utility system affects build targeting, trade requests,
    offer construction, offer acceptance, and public-pool contributions.
    """

    build_names: tuple[BuildName, ...] = (
        "infrastructure",
        "production_site",
        "advanced_site",
        "innovation",
    )
    crown_relevant_builds: tuple[BuildName, ...] = ("innovation", "infrastructure")

    def __init__(
        self,
        weights: UtilityWeights,
        *,
        build_mode: str = BUILD_MODE_DEVELOPMENT_ORIENTED,
        reserve_margin: int = 0,
        max_request_units: int = 2,
        max_payment_units: int = 4,
        min_offer_delta: float = 0.0,
        min_accept_delta: float = 0.0,
    ):
        if build_mode not in BUILD_MODE_CHOICES:
            raise ValueError(
                "build_mode must be one of "
                f"{sorted(BUILD_MODE_CHOICES)}. Got {build_mode!r}."
            )
        if max_request_units <= 0:
            raise ValueError("max_request_units must be positive.")
        if max_payment_units <= 0:
            raise ValueError("max_payment_units must be positive.")

        self.weights = weights
        self.build_mode = build_mode
        self.reserve_margin = reserve_margin
        self.max_request_units = max_request_units
        self.max_payment_units = max_payment_units
        self.min_offer_delta = min_offer_delta
        self.min_accept_delta = min_accept_delta

    # ------------------------------------------------------------------
    # Core heuristics
    # ------------------------------------------------------------------

    def _public_scores(self, state: GameState) -> tuple[int, int, int, float]:
        scores = [agent_state.score for agent_state in state.agents]
        max_score = max(scores)
        min_score = min(scores)
        score_gap = max_score - min_score
        mean_score = sum(scores) / len(scores)
        return min_score, max_score, score_gap, mean_score

    def _structurally_feasible_builds(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> list[BuildName]:
        agent_state = state.agent_state(agent_id)
        return [
            build_name
            for build_name in self.build_names
            if build_rules.has_structural_prerequisites(agent_state, build_name)
        ]

    def _continuous_site_progress(self, infrastructure: int) -> float:
        return infrastructure / 2.0

    def _innovation_points_if_built(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> int:
        agent_state = state.agent_state(agent_id)
        if agent_state.innovation >= build_rules.max_scoring_innovation:
            return 0
        return build_rules.get_project("innovation").points

    def _direct_points_if_built(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        build_name: BuildName,
    ) -> int:
        if build_name == "innovation":
            return self._innovation_points_if_built(agent_id, state, build_rules)
        return build_rules.get_project(build_name).points

    def _crown_build_bonus(
        self,
        agent_id: int,
        state: GameState,
        build_name: BuildName,
    ) -> float:
        if self.build_mode != BUILD_MODE_CROWN_AWARE:
            return 0.0
        if build_name not in self.crown_relevant_builds:
            return 0.0

        agent_state = state.agent_state(agent_id)
        if build_name == "infrastructure":
            threshold = state.infrastructure_leader_threshold
            leader_id = state.infrastructure_leader_id
            current_count = agent_state.infrastructure
            if leader_id == agent_id:
                return self.weights.crown * 0.8
            if leader_id is None:
                return self.weights.crown * max(0.0, 1.2 - 0.2 * max(0, threshold - current_count))
            leader_count = state.agent_state(leader_id).infrastructure
            return self.weights.crown * max(0.0, 1.4 - 0.2 * max(0, leader_count + 1 - current_count))

        threshold = state.innovation_leader_threshold
        leader_id = state.innovation_leader_id
        current_count = agent_state.innovation
        if leader_id == agent_id:
            return self.weights.crown * 0.8
        if leader_id is None:
            return self.weights.crown * max(0.0, 1.2 - 0.25 * max(0, threshold - current_count))
        leader_count = state.agent_state(leader_id).innovation
        return self.weights.crown * max(0.0, 1.4 - 0.25 * max(0, leader_count + 1 - current_count))

    def _build_intrinsic_value(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        build_name: BuildName,
    ) -> float:
        agent_state = state.agent_state(agent_id)
        points = self._direct_points_if_built(agent_id, state, build_rules, build_name)
        min_score, max_score, score_gap, mean_score = self._public_scores(state)

        own_score_term = self.weights.own_score * points
        social_term = self.weights.social_welfare * points
        relative_term = self.weights.relative_advantage * points
        if score_gap > 0:
            equity_scale = (max_score - agent_state.score) / score_gap
        else:
            equity_scale = 0.5
        equity_term = self.weights.equity * equity_scale * points

        capacity_term = 0.0
        site_progress_term = 0.0
        innovation_term = 0.0

        if build_name == "infrastructure":
            # Each infrastructure unit contributes half a site slot in the long run.
            before_progress = self._continuous_site_progress(agent_state.infrastructure)
            after_progress = self._continuous_site_progress(agent_state.infrastructure + 1)
            site_progress_term += self.weights.site_progress * (after_progress - before_progress)
            if build_rules.free_site_capacity(agent_state) <= 0:
                site_progress_term += 0.5 * self.weights.site_progress

        elif build_name == "production_site":
            capacity_term += 1.0 * self.weights.productive_capacity
            site_progress_term += 0.4 * self.weights.site_progress

        elif build_name == "advanced_site":
            # Upgrading one production site to one advanced site adds +1 draw net.
            capacity_term += 1.25 * self.weights.productive_capacity
            site_progress_term += 0.2 * self.weights.site_progress

        elif build_name == "innovation":
            innovation_term += self.weights.innovation_preference

        crown_term = self._crown_build_bonus(agent_id, state, build_name)
        return (
            own_score_term
            + social_term
            + relative_term
            + equity_term
            + capacity_term
            + site_progress_term
            + innovation_term
            + crown_term
        )

    def _missing_resources_for_build_with_stock(
        self,
        stock: ResourceBundle,
        build_rules: BuildRules,
        build_name: BuildName,
    ) -> dict[ResourceName, int]:
        project = build_rules.get_project(build_name)
        missing: dict[ResourceName, int] = {}
        for resource, amount in project.cost.items():
            gap = max(0, amount - stock.get(resource, 0))
            if gap > 0:
                missing[resource] = gap
        return missing

    def _target_desirability_with_stock(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        build_name: BuildName,
        stock: ResourceBundle,
    ) -> float:
        intrinsic_value = self._build_intrinsic_value(agent_id, state, build_rules, build_name)
        missing = self._missing_resources_for_build_with_stock(stock, build_rules, build_name)
        missing_units = sum(missing.values())
        target_value = intrinsic_value - self.weights.progress * missing_units

        # Affordability bonus encourages finishing projects when possible.
        if missing_units == 0:
            target_value += 0.4 * intrinsic_value + 0.5

        return target_value

    def _best_target_name_with_stock(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        stock: ResourceBundle,
    ) -> BuildName:
        candidates = self._structurally_feasible_builds(agent_id, state, build_rules)
        if not candidates:
            return "infrastructure"
        return max(
            candidates,
            key=lambda build_name: (
                self._target_desirability_with_stock(
                    agent_id=agent_id,
                    state=state,
                    build_rules=build_rules,
                    build_name=build_name,
                    stock=stock,
                ),
                -self.build_names.index(build_name),
            ),
        )

    def _best_target_name(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> BuildName:
        return self._best_target_name_with_stock(
            agent_id=agent_id,
            state=state,
            build_rules=build_rules,
            stock=state.agent_state(agent_id).stock,
        )

    def _target_cost(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        target_name: BuildName | None = None,
    ) -> dict[ResourceName, int]:
        if target_name is None:
            target_name = self._best_target_name(agent_id, state, build_rules)
        return build_rules.get_project(target_name).cost

    def _missing_for_target(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        target_name: BuildName | None = None,
        stock: ResourceBundle | None = None,
    ) -> dict[ResourceName, int]:
        if target_name is None:
            target_name = self._best_target_name(agent_id, state, build_rules)
        if stock is None:
            stock = state.agent_state(agent_id).stock
        return self._missing_resources_for_build_with_stock(stock, build_rules, target_name)

    def _reserve_cost_for_actor(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        stock: ResourceBundle | None = None,
    ) -> dict[ResourceName, int]:
        target_name = self._best_target_name_with_stock(
            agent_id=agent_id,
            state=state,
            build_rules=build_rules,
            stock=state.agent_state(agent_id).stock if stock is None else stock,
        )
        return self._target_cost(agent_id, state, build_rules, target_name)

    def _stock_after_trade(
        self,
        stock: ResourceBundle,
        received_bundle: ResourceBundle,
        paid_bundle: ResourceBundle,
    ) -> ResourceBundle | None:
        new_stock = dict(stock)
        for resource, quantity in paid_bundle.items():
            if new_stock.get(resource, 0) < quantity:
                return None
            new_stock[resource] -= quantity
        for resource, quantity in received_bundle.items():
            new_stock[resource] = new_stock.get(resource, 0) + quantity
        return new_stock

    def _stock_security_value(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        stock: ResourceBundle,
    ) -> float:
        reserve_cost = self._reserve_cost_for_actor(agent_id, state, build_rules, stock)
        protected_stock = sum(
            min(stock.get(resource, 0), reserve_cost.get(resource, 0) + self.reserve_margin)
            for resource in RESOURCES
        )
        total_stock = sum(stock.values())
        return self.weights.security * (0.6 * protected_stock + 0.2 * total_stock)

    def _stock_position_utility(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        stock: ResourceBundle,
    ) -> float:
        best_target = self._best_target_name_with_stock(agent_id, state, build_rules, stock)
        target_value = self._target_desirability_with_stock(
            agent_id=agent_id,
            state=state,
            build_rules=build_rules,
            build_name=best_target,
            stock=stock,
        )
        security_value = self._stock_security_value(agent_id, state, build_rules, stock)
        return target_value + security_value

    def _trade_delta_for_actor(
        self,
        agent_id: int,
        received_bundle: ResourceBundle,
        paid_bundle: ResourceBundle,
        state: GameState,
        build_rules: BuildRules,
    ) -> float:
        current_stock = state.agent_state(agent_id).stock
        new_stock = self._stock_after_trade(current_stock, received_bundle, paid_bundle)
        if new_stock is None:
            return -math.inf
        before_value = self._stock_position_utility(agent_id, state, build_rules, current_stock)
        after_value = self._stock_position_utility(agent_id, state, build_rules, new_stock)
        return after_value - before_value

    def _bundle_helps_target(
        self,
        agent_id: int,
        bundle: ResourceBundle,
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        missing = self._missing_for_target(agent_id, state, build_rules)
        return any(missing.get(resource, 0) > 0 and quantity > 0 for resource, quantity in bundle.items())

    def _surplus_resources_beyond_reserve(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> dict[ResourceName, int]:
        stock = state.agent_state(agent_id).stock
        reserve_cost = self._reserve_cost_for_actor(agent_id, state, build_rules)
        surplus: dict[ResourceName, int] = {}
        for resource in RESOURCES:
            reserve = reserve_cost.get(resource, 0) + self.reserve_margin
            spare = stock.get(resource, 0) - reserve
            if spare > 0:
                surplus[resource] = spare
        return surplus

    def _payment_candidates(
        self,
        agent_id: int,
        requester_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> list[ResourceName]:
        requester_stock = state.agent_state(requester_id).stock
        own_missing = self._missing_for_target(agent_id, state, build_rules)
        needed = [resource for resource, amount in own_missing.items() if amount > 0 and requester_stock.get(resource, 0) > 0]
        abundant = sorted(
            [resource for resource in RESOURCES if requester_stock.get(resource, 0) > 0 and resource not in needed],
            key=lambda resource: (-requester_stock.get(resource, 0), resource),
        )
        return needed + abundant

    def _pool_contribution_from_surplus(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
        max_units: int,
    ) -> ResourceBundle:
        surplus = self._surplus_resources_beyond_reserve(agent_id, state, build_rules)
        contribution: ResourceBundle = {}
        remaining = max_units
        for resource in sorted(surplus, key=lambda resource: (-surplus[resource], resource)):
            if remaining <= 0:
                break
            quantity = min(surplus[resource], remaining)
            if quantity > 0:
                contribution[resource] = quantity
                remaining -= quantity
        return contribution

    # ------------------------------------------------------------------
    # AgentPolicy implementation
    # ------------------------------------------------------------------

    def choose_build_target(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> BuildTarget:
        return BuildTarget(
            agent_id=agent_id,
            build_name=self._best_target_name(agent_id, state, build_rules),
        )

    def choose_build_action(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> BuildDecision:
        agent_state = state.agent_state(agent_id)
        available = build_rules.available_builds(agent_state)
        if not available:
            return NoBuildAction(agent_id=agent_id)

        build_scores = {
            build_name: self._target_desirability_with_stock(
                agent_id=agent_id,
                state=state,
                build_rules=build_rules,
                build_name=build_name,
                stock=agent_state.stock,
            )
            for build_name in available
        }
        best_build = max(
            available,
            key=lambda build_name: (build_scores[build_name], -self.build_names.index(build_name)),
        )
        if build_scores[best_build] <= 0:
            return NoBuildAction(agent_id=agent_id)
        return BuildAction(agent_id=agent_id, build_name=best_build)

    def create_trade_requests(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> list[TradeRequest]:
        target_name = self._best_target_name(agent_id, state, build_rules)
        missing = self._missing_for_target(agent_id, state, build_rules, target_name)
        if not missing:
            return []

        requested_resource = max(
            missing,
            key=lambda resource: (missing[resource], build_rules.get_project(target_name).cost.get(resource, 0), resource),
        )
        quantity = min(missing[requested_resource], self.max_request_units)
        if quantity <= 0:
            return []
        return [
            TradeRequest(
                requester_id=agent_id,
                requested_resource=requested_resource,
                quantity=quantity,
            )
        ]

    def respond_to_trade_request(
        self,
        agent_id: int,
        request: TradeRequest,
        state: GameState,
        build_rules: BuildRules,
    ) -> TradeOffer | None:
        if request.requester_id == agent_id:
            return None

        surplus = self._surplus_resources_beyond_reserve(agent_id, state, build_rules)
        available_quantity = surplus.get(request.requested_resource, 0)
        offered_quantity = min(request.quantity, available_quantity, self.max_request_units)
        if offered_quantity <= 0:
            return None

        requester_state = state.agent_state(request.requester_id)
        payment_candidates = self._payment_candidates(agent_id, request.requester_id, state, build_rules)

        best_offer: TradeOffer | None = None
        best_delta = self.min_offer_delta
        for payment_resource in payment_candidates:
            max_affordable = min(self.max_payment_units, requester_state.stock.get(payment_resource, 0))
            for payment_quantity in range(offered_quantity, max_affordable + 1):
                received_bundle = {payment_resource: payment_quantity}
                paid_bundle = {request.requested_resource: offered_quantity}
                delta = self._trade_delta_for_actor(
                    agent_id=agent_id,
                    received_bundle=received_bundle,
                    paid_bundle=paid_bundle,
                    state=state,
                    build_rules=build_rules,
                )
                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_offer = TradeOffer(
                        responder_id=agent_id,
                        requester_id=request.requester_id,
                        offered_bundle={request.requested_resource: offered_quantity},
                        requested_bundle={payment_resource: payment_quantity},
                    )

        return best_offer

    def accepts_trade_offer(
        self,
        agent_id: int,
        offer: TradeOffer,
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        if offer.requester_id != agent_id:
            return False

        delta = self._trade_delta_for_actor(
            agent_id=agent_id,
            received_bundle=offer.offered_bundle,
            paid_bundle=offer.requested_bundle,
            state=state,
            build_rules=build_rules,
        )
        return delta >= self.min_accept_delta

    def choose_pool_contribution(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> dict[str, int] | None:
        contribution_propensity = self.weights.social_welfare + self.weights.equity - 0.5 * self.weights.security
        if contribution_propensity < -0.25:
            return {}
        if contribution_propensity <= 0.10:
            return None
        max_units = max(1, min(3, int(round(contribution_propensity * 2.0))))
        contribution = self._pool_contribution_from_surplus(agent_id, state, build_rules, max_units=max_units)
        return contribution

    def accepts_pool_allocation(
        self,
        agent_id: int,
        bundle: dict[str, int],
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        if not bundle:
            return False
        delta = self._trade_delta_for_actor(
            agent_id=agent_id,
            received_bundle=bundle,
            paid_bundle={},
            state=state,
            build_rules=build_rules,
        )
        return delta >= 0 or self._bundle_helps_target(agent_id, bundle, state, build_rules)

    def trade_priority_weight(
        self,
        agent_id: int,
        state: GameState,
        build_rules: BuildRules,
    ) -> float:
        _, max_score, score_gap, _ = self._public_scores(state)
        own_score = state.agent_state(agent_id).score
        if score_gap <= 0:
            equity_component = 1.0
        else:
            equity_component = 1.0 + 0.25 * self.weights.equity * (max_score - own_score) / score_gap
        cooperation_component = 1.0 + 0.10 * self.weights.social_welfare
        return max(0.5, equity_component * cooperation_component)


# ----------------------------------------------------------------------
# Named population wrappers
# ----------------------------------------------------------------------


def _development_weights() -> UtilityWeights:
    return UtilityWeights(
        own_score=1.5,
        social_welfare=0.2,
        relative_advantage=0.2,
        equity=0.0,
        productive_capacity=2.0,
        site_progress=2.4,
        innovation_preference=0.1,
        security=0.25,
        progress=1.0,
        crown=0.3,
    )


def _cooperative_weights() -> UtilityWeights:
    return UtilityWeights(
        own_score=1.0,
        social_welfare=1.0,
        relative_advantage=0.0,
        equity=0.2,
        productive_capacity=1.8,
        site_progress=2.0,
        innovation_preference=0.3,
        security=0.15,
        progress=0.9,
        crown=0.2,
    )


def _selfish_weights() -> UtilityWeights:
    return UtilityWeights(
        own_score=1.6,
        social_welfare=0.0,
        relative_advantage=0.3,
        equity=0.0,
        productive_capacity=1.8,
        site_progress=2.0,
        innovation_preference=0.15,
        security=0.35,
        progress=1.1,
        crown=0.3,
    )


def _hoarding_weights() -> UtilityWeights:
    return UtilityWeights(
        own_score=1.1,
        social_welfare=0.0,
        relative_advantage=0.1,
        equity=0.0,
        productive_capacity=1.5,
        site_progress=1.7,
        innovation_preference=0.0,
        security=1.0,
        progress=1.2,
        crown=0.15,
    )


def _competitive_weights() -> UtilityWeights:
    return UtilityWeights(
        own_score=1.5,
        social_welfare=-0.1,
        relative_advantage=1.0,
        equity=0.0,
        productive_capacity=1.7,
        site_progress=1.9,
        innovation_preference=0.2,
        security=0.4,
        progress=1.0,
        crown=0.4,
    )


def _fairness_weights() -> UtilityWeights:
    return UtilityWeights(
        own_score=1.0,
        social_welfare=0.4,
        relative_advantage=0.0,
        equity=1.0,
        productive_capacity=1.7,
        site_progress=1.8,
        innovation_preference=0.15,
        security=0.25,
        progress=0.95,
        crown=0.15,
    )


class NeedBasedTradePolicy(UtilityBasedTradePolicy):
    def __init__(self, build_mode: str = BUILD_MODE_DEVELOPMENT_ORIENTED):
        super().__init__(
            weights=_development_weights(),
            build_mode=build_mode,
            reserve_margin=0,
            max_request_units=2,
            max_payment_units=4,
            min_offer_delta=0.0,
            min_accept_delta=-0.05,
        )


class CooperativeTradePolicy(UtilityBasedTradePolicy):
    def __init__(self, build_mode: str = BUILD_MODE_DEVELOPMENT_ORIENTED):
        super().__init__(
            weights=_cooperative_weights(),
            build_mode=build_mode,
            reserve_margin=0,
            max_request_units=2,
            max_payment_units=4,
            min_offer_delta=-0.10,
            min_accept_delta=-0.10,
        )


class SelfishTradePolicy(UtilityBasedTradePolicy):
    def __init__(self, build_mode: str = BUILD_MODE_DEVELOPMENT_ORIENTED):
        super().__init__(
            weights=_selfish_weights(),
            build_mode=build_mode,
            reserve_margin=1,
            max_request_units=2,
            max_payment_units=4,
            min_offer_delta=0.05,
            min_accept_delta=0.0,
        )


class HoardingTradePolicy(UtilityBasedTradePolicy):
    def __init__(self, build_mode: str = BUILD_MODE_DEVELOPMENT_ORIENTED):
        super().__init__(
            weights=_hoarding_weights(),
            build_mode=build_mode,
            reserve_margin=2,
            max_request_units=2,
            max_payment_units=5,
            min_offer_delta=0.25,
            min_accept_delta=0.10,
        )


class CompetitiveTradePolicy(UtilityBasedTradePolicy):
    def __init__(self, build_mode: str = BUILD_MODE_DEVELOPMENT_ORIENTED):
        super().__init__(
            weights=_competitive_weights(),
            build_mode=build_mode,
            reserve_margin=1,
            max_request_units=2,
            max_payment_units=4,
            min_offer_delta=0.10,
            min_accept_delta=0.05,
        )

    def accepts_trade_offer(
        self,
        agent_id: int,
        offer: TradeOffer,
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        if offer.requester_id != agent_id:
            return False
        responder_score = state.agent_state(offer.responder_id).score
        own_score = state.agent_state(agent_id).score
        delta = self._trade_delta_for_actor(
            agent_id=agent_id,
            received_bundle=offer.offered_bundle,
            paid_bundle=offer.requested_bundle,
            state=state,
            build_rules=build_rules,
        )
        if responder_score > own_score and delta < self.min_accept_delta + 0.10:
            return False
        return delta >= self.min_accept_delta


class FairnessSensitiveTradePolicy(UtilityBasedTradePolicy):
    def __init__(self, build_mode: str = BUILD_MODE_DEVELOPMENT_ORIENTED):
        super().__init__(
            weights=_fairness_weights(),
            build_mode=build_mode,
            reserve_margin=0,
            max_request_units=2,
            max_payment_units=4,
            min_offer_delta=-0.05,
            min_accept_delta=-0.05,
        )

    def accepts_trade_offer(
        self,
        agent_id: int,
        offer: TradeOffer,
        state: GameState,
        build_rules: BuildRules,
    ) -> bool:
        if offer.requester_id != agent_id:
            return False
        own_score = state.agent_state(agent_id).score
        responder_score = state.agent_state(offer.responder_id).score
        delta = self._trade_delta_for_actor(
            agent_id=agent_id,
            received_bundle=offer.offered_bundle,
            paid_bundle=offer.requested_bundle,
            state=state,
            build_rules=build_rules,
        )
        fairness_bonus = 0.05 if responder_score < own_score else 0.0
        fairness_penalty = 0.10 if responder_score > own_score else 0.0
        return delta >= self.min_accept_delta - fairness_bonus + fairness_penalty
