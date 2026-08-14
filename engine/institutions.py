"""
Exchange institutions.

An institution is the layer between public intentions and actual cross-agent
resource transfers. It asks policies for requests or offers, selects what is
allowed to happen, and delegates the real mutation back to `Game`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from engine.actions import ResourceBundle, TradeOffer, TradeRequest
from engine.resources import RESOURCES, ResourceName
from engine.state import GameState


def _agent_target_and_missing(
    game,
    state: GameState,
    agent_id: int,
) -> tuple[str, dict[str, int]]:
    """Return an agent's current build target and resource shortfall."""
    policy = game.policies[agent_id]
    target = policy.choose_build_target(
        agent_id=agent_id,
        state=state,
        build_rules=game.build_rules,
    )
    agent_state = state.agent_state(agent_id)
    missing = game.build_rules.missing_resources_for_build(
        agent_state,
        target.build_name,
    )
    return target.build_name, missing


def _total_missing_units(missing: dict[str, int]) -> int:
    """Collapse a resource shortfall bundle to its total number of units."""
    return sum(missing.values())


def _sanitize_bundle(
    bundle: dict[str, int] | ResourceBundle | None,
) -> ResourceBundle:
    """Convert quantities to positive integers and discard empty entries."""
    if bundle is None:
        return {}

    return {
        resource: int(quantity)
        for resource, quantity in bundle.items()
        if int(quantity) > 0
    }


def _refresh_trade_request(
    game,
    state: GameState,
    request: TradeRequest,
) -> TradeRequest | None:
    """Revalidate a request against the requester's current target shortage.

    A requester may have received the resource earlier in the same institution
    phase, for example as payment while responding to another request. This
    helper prevents stale requests from being fulfilled after the underlying
    shortage has disappeared and caps the quantity at the remaining shortage.
    """
    _, missing = _agent_target_and_missing(game, state, request.requester_id)
    remaining = missing.get(request.requested_resource, 0)
    if remaining <= 0:
        return None

    return TradeRequest(
        requester_id=request.requester_id,
        requested_resource=request.requested_resource,
        quantity=min(request.quantity, remaining),
    )


def _collect_trade_requests(
    game,
    state: GameState,
    max_requests_per_agent: int = 1,
) -> list[TradeRequest]:
    """Collect a bounded number of policy-generated requests from each agent."""
    requests = []

    for agent_state in state.agents:
        policy = game.policies[agent_state.id]
        agent_requests = policy.create_trade_requests(
            agent_id=agent_state.id,
            state=state,
            build_rules=game.build_rules,
        )
        requests.extend(agent_requests[:max_requests_per_agent])

    return requests


def _resource_reserve_for_target(
    game,
    state: GameState,
    agent_id: int,
    resource: ResourceName,
) -> int:
    """Return how many units of one resource an agent's target reserves."""
    policy = game.policies[agent_id]
    target = policy.choose_build_target(
        agent_id=agent_id,
        state=state,
        build_rules=game.build_rules,
    )
    project = game.build_rules.get_project(target.build_name)
    return project.cost.get(resource, 0)


def _can_spare_resource(
    game,
    state: GameState,
    agent_id: int,
    resource: ResourceName,
    quantity: int = 1,
) -> bool:
    """Check whether stock exceeds the current target reserve by ``quantity``."""
    agent_state = state.agent_state(agent_id)
    reserve = _resource_reserve_for_target(game, state, agent_id, resource)
    owned = agent_state.stock.get(resource, 0)
    return owned - reserve >= quantity


def _can_spare_bundle_for_target(
    game,
    state: GameState,
    agent_id: int,
    bundle: ResourceBundle,
) -> bool:
    """Check whether paying a bundle preserves the agent's current target reserve."""
    agent_state = state.agent_state(agent_id)
    for resource, quantity in bundle.items():
        reserve = _resource_reserve_for_target(game, state, agent_id, resource)
        if agent_state.stock.get(resource, 0) - quantity < reserve:
            return False
    return True


def _choose_payment_resource(
    game,
    state: GameState,
    requester_id: int,
    supplier_id: int,
) -> ResourceName | None:
    """Choose a target-safe one-unit payment resource.

    Payments that reduce the supplier's current target shortage are preferred.
    The requester must have genuine surplus above the complete cost of its own
    current target; comparing against the requester's *missing* amount would
    incorrectly classify exactly-reserved resources as expendable.
    """
    requester_state = state.agent_state(requester_id)
    _, supplier_missing = _agent_target_and_missing(game, state, supplier_id)

    target_safe_resources = [
        resource
        for resource in RESOURCES
        if requester_state.stock.get(resource, 0) > 0
        and _can_spare_resource(
            game,
            state,
            requester_id,
            resource,
            quantity=1,
        )
    ]
    if not target_safe_resources:
        return None

    preferred_resources = [
        resource
        for resource in target_safe_resources
        if supplier_missing.get(resource, 0) > 0
    ]
    if preferred_resources:
        return max(
            preferred_resources,
            key=lambda resource: (
                supplier_missing[resource],
                requester_state.stock.get(resource, 0),
                resource,
            ),
        )

    return max(
        target_safe_resources,
        key=lambda resource: (
            requester_state.stock.get(resource, 0)
            - _resource_reserve_for_target(
                game, state, requester_id, resource
            ),
            requester_state.stock.get(resource, 0),
            resource,
        ),
    )


def _default_pool_contribution(
    game,
    state: GameState,
    agent_id: int,
) -> ResourceBundle:
    """
    Conservative fallback for public-pool institutions.

    Contribute only clear surplus above the current target reserve, and also
    preserve enough stock to avoid blocking any immediately affordable build.
    """
    agent_state = state.agent_state(agent_id)
    policy = game.policies[agent_id]
    target = policy.choose_build_target(
        agent_id=agent_id,
        state=state,
        build_rules=game.build_rules,
    )
    target_cost = game.build_rules.get_project(target.build_name).cost

    reserve = {
        resource: target_cost.get(resource, 0)
        for resource in RESOURCES
    }

    for build_name in game.build_rules.available_builds(agent_state):
        build_cost = game.build_rules.get_project(build_name).cost
        for resource, amount in build_cost.items():
            reserve[resource] = max(reserve[resource], amount)

    return {
        resource: agent_state.stock.get(resource, 0) - reserve[resource]
        for resource in RESOURCES
        if agent_state.stock.get(resource, 0) - reserve[resource] > 0
    }


@dataclass
class InstitutionResult:
    """
    Summary of what the institution did this round.

    For NoTradeInstitution, everything is zero/empty.
    Later, bilateral trade and central markets can fill this with:
    - number of proposed trades
    - number of accepted trades
    - executed trades
    - rejected trades
    """
    institution_name: str
    trades_proposed: int = 0
    trades_executed: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class Institution(ABC):
    """
    Base class for all trade/allocation institutions.

    Institutions control the social organization of exchange or allocation.

    Policies determine what agents want, offer, accept, or contribute.
    Institutions determine:
    - what information is collected
    - how requests or offers are ordered
    - whether matching is bilateral or centralized
    - whether transfers require voluntary exchange
    - how many trades or allocations can happen in a round
    """

    name: str

    @abstractmethod
    def resolve(
        self,
        game,
        state: GameState,
    ) -> InstitutionResult:
        """Resolve one institution phase and return event counts plus details."""
        pass


class NoTradeInstitution(Institution):
    """
    Baseline institution: no exchange is allowed.

    Control:
    - ignores agent-generated offers and requests
    - does not construct trades centrally
    - does not transfer resources without voluntary exchange
    """

    name = "no_trade"

    def resolve(
        self,
        game,
        state: GameState,
    ) -> InstitutionResult:
        """Return an explicit zero-activity result for the baseline condition."""
        return InstitutionResult(
            institution_name=self.name,
            trades_proposed=0,
            trades_executed=0,
        )


class BilateralTradeInstitution(Institution):
    """
    First bounded bilateral bargaining institution.

    The protocol is intentionally narrow: one request per agent, one offer per
    responder, at most one accepted offer per request, and no counteroffers.

    Control:
    - uses agent-generated requests and agent-generated offers
    - does not construct barter terms centrally
    - resolves accepted offers request-by-request rather than globally
    - does not transfer resources without voluntary exchange
    """

    name = "bilateral_trade"

    def __init__(
        self,
        max_requests_per_agent: int = 1,
        shuffle_requests: bool = True,
        max_bargaining_passes: int = 1,
    ):
        """Configure request limits, random ordering, and bargaining passes."""
        if max_requests_per_agent <= 0:
            raise ValueError("max_requests_per_agent must be positive.")
        if max_bargaining_passes <= 0:
            raise ValueError("max_bargaining_passes must be positive.")

        self.max_requests_per_agent = max_requests_per_agent
        self.shuffle_requests = shuffle_requests
        self.max_bargaining_passes = max_bargaining_passes

    def _offer_total_payment(self, offer: TradeOffer) -> int:
        """Return the number of resource units requested as payment."""
        return sum(offer.requested_bundle.values())

    def _offer_details(self, offer: TradeOffer) -> dict[str, Any]:
        """Convert an executed offer into a CSV-friendly audit record."""
        return {
            "responder_id": offer.responder_id,
            "requester_id": offer.requester_id,
            "offered_bundle": dict(offer.offered_bundle),
            "requested_bundle": dict(offer.requested_bundle),
        }

    def _ordered_requests(
        self,
        requests: list[TradeRequest],
        game,
        state: GameState,
    ) -> list[TradeRequest]:
        """Return request order, optionally shuffled with the game RNG."""
        ordered_requests = list(requests)

        if self.shuffle_requests and ordered_requests:
            game.institution_rng.shuffle(ordered_requests)

        return ordered_requests

    def _run_single_pass(
        self,
        game,
        state: GameState,
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Run one request-offer-accept-execute bargaining pass.

        Requests are handled sequentially. State is refreshed after each
        execution so later responders cannot offer resources already spent.
        """
        requests = _collect_trade_requests(
            game=game,
            state=state,
            max_requests_per_agent=self.max_requests_per_agent,
        )

        requests = self._ordered_requests(
            requests=requests,
            game=game,
            state=state,
        )

        trades_proposed = 0
        trades_executed = 0
        executed_offers: list[dict[str, Any]] = []
        current_state = state

        for request in requests:
            current_request = _refresh_trade_request(
                game=game,
                state=current_state,
                request=request,
            )
            if current_request is None:
                continue

            acceptable_offers: list[TradeOffer] = []

            for responder_state in current_state.agents:
                if responder_state.id == current_request.requester_id:
                    continue

                policy = game.policies[responder_state.id]
                offer = policy.respond_to_trade_request(
                    agent_id=responder_state.id,
                    request=current_request,
                    state=current_state,
                    build_rules=game.build_rules,
                )

                if offer is None:
                    continue

                trades_proposed += 1

                if not game.can_pay_bundle(offer.responder_id, offer.offered_bundle):
                    continue

                if not game.can_pay_bundle(offer.requester_id, offer.requested_bundle):
                    continue

                requester_policy = game.policies[current_request.requester_id]
                if requester_policy.accepts_trade_offer(
                    agent_id=current_request.requester_id,
                    offer=offer,
                    state=current_state,
                    build_rules=game.build_rules,
                ):
                    acceptable_offers.append(offer)

            if not acceptable_offers:
                continue

            lowest_payment = min(
                self._offer_total_payment(offer)
                for offer in acceptable_offers
            )
            cheapest_offers = [
                offer
                for offer in acceptable_offers
                if self._offer_total_payment(offer) == lowest_payment
            ]

            chosen_offer = cheapest_offers[0]
            if len(cheapest_offers) > 1:
                chosen_offer = game.institution_rng.choice(cheapest_offers)

            if game.execute_trade_offer(chosen_offer):
                trades_executed += 1
                executed_offers.append(self._offer_details(chosen_offer))
                current_state = game.get_state()

        return trades_proposed, trades_executed, executed_offers

    def resolve(
        self,
        game,
        state: GameState,
    ) -> InstitutionResult:
        """Run bounded bargaining passes until the limit or a no-trade pass."""
        current_state = state
        total_trades_proposed = 0
        total_trades_executed = 0
        all_executed_offers: list[dict[str, Any]] = []

        for _ in range(self.max_bargaining_passes):
            trades_proposed, trades_executed, executed_offers = self._run_single_pass(
                game=game,
                state=current_state,
            )
            total_trades_proposed += trades_proposed
            total_trades_executed += trades_executed
            all_executed_offers.extend(executed_offers)

            if trades_executed == 0:
                break

            current_state = game.get_state()

        return InstitutionResult(
            institution_name=self.name,
            trades_proposed=total_trades_proposed,
            trades_executed=total_trades_executed,
            details={"executed_offers": all_executed_offers},
        )


class CatchUpBilateralTradeInstitution(BilateralTradeInstitution):
    """
    Bilateral trade with request priority for lower-scoring agents.

    Control:
    - still uses agent-generated offers
    - keeps bilateral voluntary exchange
    - changes only request ordering, prioritizing lower-score requesters
    """

    name = "catch_up_bilateral_trade"

    def _ordered_requests(
        self,
        requests: list[TradeRequest],
        game,
        state: GameState,
    ) -> list[TradeRequest]:
        """Prioritize lower-scoring requesters, with optional random tie breaks."""
        tie_breakers = {
            id(request): game.institution_rng.random() if self.shuffle_requests else 0.0
            for request in requests
        }

        return sorted(
            requests,
            key=lambda request: (
                state.agent_state(request.requester_id).score,
                tie_breakers[id(request)],
            ),
        )


class BottleneckPriorityBilateralTradeInstitution(BilateralTradeInstitution):
    """
    Bilateral trade with priority for requests closest to unlocking development.

    Control:
    - still uses agent-generated offers
    - keeps bilateral voluntary exchange
    - changes only request ordering, prioritizing bottleneck-reducing requests
    """

    name = "bottleneck_priority_bilateral_trade"

    def _request_priority(
        self,
        request: TradeRequest,
        game,
        state: GameState,
    ) -> tuple[int, int, int]:
        """Rank requests by remaining target shortfall after fulfillment."""
        requester_state = state.agent_state(request.requester_id)
        requester_policy = game.policies[request.requester_id]
        target = requester_policy.choose_build_target(
            agent_id=request.requester_id,
            state=state,
            build_rules=game.build_rules,
        )
        missing = game.build_rules.missing_resources_for_build(
            requester_state,
            target.build_name,
        )

        missing_before = sum(missing.values())
        current_gap = missing.get(request.requested_resource, 0)
        missing_reduction = min(current_gap, request.quantity)
        missing_after = missing_before - missing_reduction

        return (
            missing_after,
            -missing_reduction,
            requester_state.score,
        )

    def _ordered_requests(
        self,
        requests: list[TradeRequest],
        game,
        state: GameState,
    ) -> list[TradeRequest]:
        """Sort requests by bottleneck reduction, score, and tie breaker."""
        tie_breakers = {
            id(request): game.institution_rng.random() if self.shuffle_requests else 0.0
            for request in requests
        }

        return sorted(
            requests,
            key=lambda request: (
                *self._request_priority(request, game, state),
                tie_breakers[id(request)],
            ),
        )


class SubsidizedCatchUpInstitution(Institution):
    """
    Redistributive support rule for low-scoring agents one unit short of a build.

    Control:
    - does not use agent-generated offers
    - does not construct market-like barter terms
    - can transfer resources without reciprocal voluntary exchange
    """

    name = "subsidized_catch_up"

    def __init__(self, max_subsidies_per_round: int = 2):
        """Set the maximum number of one-unit subsidies in a round."""
        if max_subsidies_per_round <= 0:
            raise ValueError("max_subsidies_per_round must be positive.")

        self.max_subsidies_per_round = max_subsidies_per_round

    def resolve(
        self,
        game,
        state: GameState,
    ) -> InstitutionResult:
        """Transfer missing units to low scorers without reciprocal payment.

        Only agents at or below the median score and exactly one unit short of
        their current target qualify. Donors with true target surplus are
        preferred before donors selected merely by stock size.
        """
        scores = [agent_state.score for agent_state in state.agents]
        median_score = median(scores)
        low_scoring_agents = [
            agent_state
            for agent_state in state.agents
            if agent_state.score <= median_score
        ]
        low_scoring_agents.sort(key=lambda agent_state: (agent_state.score, agent_state.id))

        trades_proposed = 0
        trades_executed = 0
        executed_subsidies: list[dict[str, Any]] = []
        current_state = state

        for requester_state in low_scoring_agents:
            if trades_executed >= self.max_subsidies_per_round:
                break

            _, missing = _agent_target_and_missing(game, current_state, requester_state.id)
            if _total_missing_units(missing) != 1:
                continue

            missing_resource = next(iter(missing))
            donor_candidates = []

            for donor_state in current_state.agents:
                if donor_state.id == requester_state.id:
                    continue

                owned = donor_state.stock.get(missing_resource, 0)
                if owned <= 1:
                    continue

                surplus_preferred = _can_spare_resource(
                    game,
                    current_state,
                    donor_state.id,
                    missing_resource,
                    quantity=1,
                )
                donor_candidates.append((not surplus_preferred, -owned, donor_state.id))

            if not donor_candidates:
                continue

            trades_proposed += 1
            donor_candidates.sort()
            donor_id = donor_candidates[0][2]

            if game.transfer_bundle(
                from_agent_id=donor_id,
                to_agent_id=requester_state.id,
                bundle={missing_resource: 1},
            ):
                trades_executed += 1
                executed_subsidies.append(
                    {
                        "requester_id": requester_state.id,
                        "donor_id": donor_id,
                        "resource": missing_resource,
                        "quantity": 1,
                    }
                )
                current_state = game.get_state()

        return InstitutionResult(
            institution_name=self.name,
            trades_proposed=trades_proposed,
            trades_executed=trades_executed,
            details={"subsidies_executed": executed_subsidies},
        )


class CentralMarketClearingInstitution(Institution):
    """
    Iterative centralized one-for-one barter matcher.

    The institution uses agent-generated requests but constructs suppliers,
    payment resources, and execution order centrally. Candidate trades are
    rebuilt after every successful execution so requests, reserves, and target
    shortages are always evaluated against the current state.
    """

    name = "central_clearing"

    def __init__(self, max_trades_per_round: int | None = None):
        """Configure an optional cap on centrally executed reciprocal trades."""
        if max_trades_per_round is not None and max_trades_per_round <= 0:
            raise ValueError("max_trades_per_round must be positive or None.")

        self.max_trades_per_round = max_trades_per_round

    def _missing_after_trade(
        self,
        game,
        state: GameState,
        agent_id: int,
        received_resource: ResourceName,
        paid_resource: ResourceName,
    ) -> int:
        """Return current-target missing units after a simulated one-for-one swap."""
        target_name, _ = _agent_target_and_missing(game, state, agent_id)
        project = game.build_rules.get_project(target_name)
        stock_after = state.agent_state(agent_id).stock.copy()
        stock_after[received_resource] = stock_after.get(received_resource, 0) + 1
        stock_after[paid_resource] = stock_after.get(paid_resource, 0) - 1
        missing_after = game.build_rules.missing_resources(stock_after, project.cost)
        return _total_missing_units(missing_after)

    def _candidate_reduction_score(
        self,
        game,
        state: GameState,
        requester_id: int,
        supplier_id: int,
        requested_resource: ResourceName,
        payment_resource: ResourceName,
    ) -> tuple[int, int]:
        """Measure the net current-target shortage reduction for both agents."""
        _, requester_missing = _agent_target_and_missing(game, state, requester_id)
        _, supplier_missing = _agent_target_and_missing(game, state, supplier_id)
        total_missing_before = (
            _total_missing_units(requester_missing)
            + _total_missing_units(supplier_missing)
        )

        requester_after = self._missing_after_trade(
            game=game,
            state=state,
            agent_id=requester_id,
            received_resource=requested_resource,
            paid_resource=payment_resource,
        )
        supplier_after = self._missing_after_trade(
            game=game,
            state=state,
            agent_id=supplier_id,
            received_resource=payment_resource,
            paid_resource=requested_resource,
        )
        total_missing_after = requester_after + supplier_after
        return total_missing_before - total_missing_after, total_missing_after

    def _candidate_priority(
        self,
        game,
        state: GameState,
        requester_id: int,
        supplier_id: int,
        requested_resource: ResourceName,
        payment_resource: ResourceName,
    ) -> tuple[float, int, int, float]:
        """Rank net productive candidates before low-score requesters and ties."""
        reduction_score, total_missing_after = self._candidate_reduction_score(
            game,
            state,
            requester_id,
            supplier_id,
            requested_resource,
            payment_resource,
        )
        requester_state = state.agent_state(requester_id)
        return (
            -float(reduction_score),
            total_missing_after,
            requester_state.score,
            game.institution_rng.random(),
        )

    def _build_candidate_trades(
        self,
        game,
        state: GameState,
    ) -> list[tuple[tuple[Any, ...], TradeOffer]]:
        """Construct valid candidates from the current state only."""
        requests = _collect_trade_requests(
            game=game,
            state=state,
            max_requests_per_agent=1,
        )
        candidate_trades: list[tuple[tuple[Any, ...], TradeOffer]] = []

        for request in requests:
            current_request = _refresh_trade_request(game, state, request)
            if current_request is None:
                continue

            for supplier_state in state.agents:
                if supplier_state.id == current_request.requester_id:
                    continue

                if not _can_spare_resource(
                    game,
                    state,
                    supplier_state.id,
                    current_request.requested_resource,
                    quantity=1,
                ):
                    continue

                payment_resource = _choose_payment_resource(
                    game,
                    state,
                    requester_id=current_request.requester_id,
                    supplier_id=supplier_state.id,
                )
                if payment_resource is None:
                    continue

                offer = TradeOffer(
                    responder_id=supplier_state.id,
                    requester_id=current_request.requester_id,
                    offered_bundle={current_request.requested_resource: 1},
                    requested_bundle={payment_resource: 1},
                )

                if not _can_spare_bundle_for_target(
                    game, state, offer.responder_id, offer.offered_bundle
                ):
                    continue
                if not _can_spare_bundle_for_target(
                    game, state, offer.requester_id, offer.requested_bundle
                ):
                    continue

                reduction_score, _ = self._candidate_reduction_score(
                    game,
                    state,
                    current_request.requester_id,
                    supplier_state.id,
                    current_request.requested_resource,
                    payment_resource,
                )
                if reduction_score <= 0:
                    continue

                candidate_trades.append(
                    (
                        self._candidate_priority(
                            game=game,
                            state=state,
                            requester_id=current_request.requester_id,
                            supplier_id=supplier_state.id,
                            requested_resource=current_request.requested_resource,
                            payment_resource=payment_resource,
                        ),
                        offer,
                    )
                )

        candidate_trades.sort(key=lambda item: item[0])
        return candidate_trades

    def resolve(
        self,
        game,
        state: GameState,
    ) -> InstitutionResult:
        """Iteratively recompute, rank, and execute currently valid candidates."""
        current_state = state
        trades_proposed = 0
        trades_executed = 0
        candidate_rebuilds = 0
        executed_offers: list[dict[str, Any]] = []

        while (
            self.max_trades_per_round is None
            or trades_executed < self.max_trades_per_round
        ):
            candidate_trades = self._build_candidate_trades(game, current_state)
            candidate_rebuilds += 1
            trades_proposed += len(candidate_trades)
            if not candidate_trades:
                break

            executed = False
            for _, offer in candidate_trades:
                if game.execute_trade_offer(offer):
                    trades_executed += 1
                    executed_offers.append(
                        {
                            "responder_id": offer.responder_id,
                            "requester_id": offer.requester_id,
                            "offered_bundle": dict(offer.offered_bundle),
                            "requested_bundle": dict(offer.requested_bundle),
                        }
                    )
                    current_state = game.get_state()
                    executed = True
                    break

            if not executed:
                break

        return InstitutionResult(
            institution_name=self.name,
            trades_proposed=trades_proposed,
            trades_executed=trades_executed,
            details={
                "executed_offers": executed_offers,
                "candidate_rebuilds": candidate_rebuilds,
            },
        )


class ClearinghouseBargainingInstitution(BilateralTradeInstitution):
    """
    Globally coordinated execution of voluntary accepted offers.

    Requests, responder offers, and requester acceptance remain policy-driven.
    The clearinghouse repeatedly recomputes accepted offers after each execution
    and allows at most one executed offer for each initial request in a pass.
    This creates a genuinely compatible subset without relying on stale stocks,
    reserves, or acceptance decisions.
    """

    name = "clearinghouse_bargaining"

    def __init__(
        self,
        max_bargaining_passes: int = 1,
        max_trades_per_round: int | None = None,
    ):
        """Configure coordinated bargaining passes and an optional round cap."""
        if max_trades_per_round is not None and max_trades_per_round <= 0:
            raise ValueError("max_trades_per_round must be positive or None.")

        super().__init__(
            max_requests_per_agent=1,
            shuffle_requests=True,
            max_bargaining_passes=max_bargaining_passes,
        )
        self.max_trades_per_round = max_trades_per_round

    def _accepted_offer_priority(
        self,
        offer: TradeOffer,
        game,
        state: GameState,
    ) -> tuple[int, int, float]:
        """Prefer cheaper accepted offers, then lower-scoring requesters."""
        requester_state = state.agent_state(offer.requester_id)
        return (
            self._offer_total_payment(offer),
            requester_state.score,
            game.institution_rng.random(),
        )

    def _collect_accepted_offers(
        self,
        game,
        state: GameState,
        pending_requests: dict[int, TradeRequest],
    ) -> tuple[set[tuple[Any, ...]], list[TradeOffer], set[int]]:
        """Recompute valid accepted offers for all still-pending requests."""
        proposal_signatures: set[tuple[Any, ...]] = set()
        accepted_offers: list[TradeOffer] = []
        stale_requesters: set[int] = set()

        for requester_id, original_request in pending_requests.items():
            request = _refresh_trade_request(game, state, original_request)
            if request is None:
                stale_requesters.add(requester_id)
                continue

            for responder_state in state.agents:
                if responder_state.id == request.requester_id:
                    continue

                policy = game.policies[responder_state.id]
                offer = policy.respond_to_trade_request(
                    agent_id=responder_state.id,
                    request=request,
                    state=state,
                    build_rules=game.build_rules,
                )
                if offer is None:
                    continue

                signature = (
                    offer.responder_id,
                    offer.requester_id,
                    tuple(sorted(offer.offered_bundle.items())),
                    tuple(sorted(offer.requested_bundle.items())),
                )
                proposal_signatures.add(signature)

                if not game.can_pay_bundle(offer.responder_id, offer.offered_bundle):
                    continue
                if not game.can_pay_bundle(offer.requester_id, offer.requested_bundle):
                    continue

                requester_policy = game.policies[request.requester_id]
                if requester_policy.accepts_trade_offer(
                    agent_id=request.requester_id,
                    offer=offer,
                    state=state,
                    build_rules=game.build_rules,
                ):
                    accepted_offers.append(offer)

        accepted_offers.sort(
            key=lambda offer: self._accepted_offer_priority(
                offer=offer, game=game, state=state
            )
        )
        return proposal_signatures, accepted_offers, stale_requesters

    def _run_pass_with_cap(
        self,
        game,
        state: GameState,
        trade_cap: int | None,
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Execute a compatible subset for one fixed set of initial requests."""
        requests = _collect_trade_requests(
            game=game,
            state=state,
            max_requests_per_agent=self.max_requests_per_agent,
        )
        requests = self._ordered_requests(requests, game, state)
        pending_requests = {request.requester_id: request for request in requests}

        unique_proposals: set[tuple[Any, ...]] = set()
        trades_executed = 0
        executed_offers: list[dict[str, Any]] = []
        current_state = state

        while pending_requests and (trade_cap is None or trades_executed < trade_cap):
            proposals, accepted_offers, stale_requesters = self._collect_accepted_offers(
                game=game,
                state=current_state,
                pending_requests=pending_requests,
            )
            unique_proposals.update(proposals)
            for requester_id in stale_requesters:
                pending_requests.pop(requester_id, None)

            if not accepted_offers:
                break

            executed = False
            for offer in accepted_offers:
                if game.execute_trade_offer(offer):
                    trades_executed += 1
                    executed_offers.append(self._offer_details(offer))
                    pending_requests.pop(offer.requester_id, None)
                    current_state = game.get_state()
                    executed = True
                    break

            if not executed:
                break

        return len(unique_proposals), trades_executed, executed_offers

    def _run_single_pass(
        self,
        game,
        state: GameState,
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Compatibility wrapper used by the inherited protocol interface."""
        return self._run_pass_with_cap(
            game=game,
            state=state,
            trade_cap=self.max_trades_per_round,
        )

    def resolve(
        self,
        game,
        state: GameState,
    ) -> InstitutionResult:
        """Run passes while enforcing the configured cap across the whole round."""
        current_state = state
        total_proposed = 0
        total_executed = 0
        all_executed: list[dict[str, Any]] = []

        for _ in range(self.max_bargaining_passes):
            remaining_cap = (
                None
                if self.max_trades_per_round is None
                else self.max_trades_per_round - total_executed
            )
            if remaining_cap is not None and remaining_cap <= 0:
                break

            proposed, executed, offers = self._run_pass_with_cap(
                game=game,
                state=current_state,
                trade_cap=remaining_cap,
            )
            total_proposed += proposed
            total_executed += executed
            all_executed.extend(offers)
            if executed == 0:
                break
            current_state = game.get_state()

        return InstitutionResult(
            institution_name=self.name,
            trades_proposed=total_proposed,
            trades_executed=total_executed,
            details={"executed_offers": all_executed},
        )


class EquityWeightedCentralClearingInstitution(CentralMarketClearingInstitution):
    """
    Equity-oriented centralized matcher.

    Control:
    - uses agent-generated requests but not agent-generated offers
    - constructs trades centrally like `CentralMarketClearingInstitution`
    - prioritizes lower-scoring requesters through a central equity-weighted
      ranking rule
    - may de-prioritize helping the current highest-scoring agent unless the
      trade is strongly productive
    """

    name = "equity_weighted_central_clearing"

    def __init__(
        self,
        equity_weight: float = 1.0,
        max_trades_per_round: int | None = None,
    ):
        """Configure the strength of equity weighting in central priority."""
        if equity_weight < 0:
            raise ValueError("equity_weight must be non-negative.")

        super().__init__(max_trades_per_round=max_trades_per_round)
        self.equity_weight = equity_weight

    def _candidate_priority(
        self,
        game,
        state: GameState,
        requester_id: int,
        supplier_id: int,
        requested_resource: ResourceName,
        payment_resource: ResourceName,
    ) -> tuple[bool, float, int, int, float]:
        """Combine shortage reduction with requester disadvantage in ranking."""
        reduction_score, total_missing_after = self._candidate_reduction_score(
            game,
            state,
            requester_id,
            supplier_id,
            requested_resource,
            payment_resource,
        )
        requester_state = state.agent_state(requester_id)
        scores = [agent_state.score for agent_state in state.agents]
        highest_score = max(scores)
        lowest_score = min(scores)
        score_span = max(1, highest_score - lowest_score)
        relative_disadvantage = (highest_score - requester_state.score) / score_span
        equity_adjusted_productivity = reduction_score + (
            self.equity_weight * relative_disadvantage
        )
        highest_score_penalty = (
            requester_state.score == highest_score and reduction_score < 2
        )

        return (
            highest_score_penalty,
            -equity_adjusted_productivity,
            total_missing_after,
            requester_state.score,
            game.institution_rng.random(),
        )


class PublicPoolInstitution(Institution):
    """
    Collective support institution based on contributions and centralized allocation.

    Control:
    - does not use agent-generated barter offers
    - does not construct reciprocal trades
    - can transfer resources without voluntary exchange once resources have
      entered the pool
    - uses policy hooks for contributions and allocation acceptance
    """

    name = "public_pool"

    def __init__(
        self,
        max_allocations_per_round: int = 3,
        prioritize_low_score: bool = True,
    ):
        """Configure allocation capacity and optional low-score priority."""
        if max_allocations_per_round <= 0:
            raise ValueError("max_allocations_per_round must be positive.")

        self.max_allocations_per_round = max_allocations_per_round
        self.prioritize_low_score = prioritize_low_score
        self.pool = {resource: 0 for resource in RESOURCES}

    def _choose_contribution(
        self,
        game,
        state: GameState,
        agent_id: int,
    ) -> ResourceBundle:
        """Respect explicit abstention and use fallback only for ``None``.

        ``None`` means that a policy has no custom contribution rule. An empty
        dictionary is an explicit decision to contribute nothing and must not
        be replaced by the fallback.
        """
        policy = game.policies[agent_id]
        agent_state = state.agent_state(agent_id)
        raw_contribution = policy.choose_pool_contribution(
            agent_id=agent_id,
            state=state,
            build_rules=game.build_rules,
        )

        if raw_contribution is None:
            return _default_pool_contribution(
                game=game,
                state=state,
                agent_id=agent_id,
            )

        explicit_contribution = _sanitize_bundle(raw_contribution)
        return {
            resource: min(quantity, agent_state.stock.get(resource, 0))
            for resource, quantity in explicit_contribution.items()
            if min(quantity, agent_state.stock.get(resource, 0)) > 0
        }

    def _allocation_priority(
        self,
        agent_state,
        missing: dict[str, int],
        bundle: ResourceBundle,
    ) -> tuple[int, int, int, int]:
        """Prefer allocations that leave least unmet need, then low scores."""
        remaining_missing = _total_missing_units(missing) - sum(bundle.values())
        score_priority = agent_state.score if self.prioritize_low_score else 0
        return (
            remaining_missing,
            score_priority,
            _total_missing_units(missing),
            agent_state.id,
        )

    def resolve(
        self,
        game,
        state: GameState,
    ) -> InstitutionResult:
        """Collect contributions, then greedily allocate target-relevant bundles.

        Contributions are removed before allocation begins. Unallocated pool
        stock remains available in later rounds of the same game and is
        reported in the result details.
        """
        contributions: list[dict[str, Any]] = []

        for agent_state in state.agents:
            contribution = self._choose_contribution(
                game=game,
                state=state,
                agent_id=agent_state.id,
            )
            if not contribution:
                continue

            if not game.can_pay_bundle(agent_state.id, contribution):
                continue

            live_agent = game.get_agent_by_id(agent_state.id)
            for resource, quantity in contribution.items():
                live_agent.stock[resource] -= quantity
                self.pool[resource] += quantity

            contributions.append(
                {
                    "agent_id": agent_state.id,
                    "bundle": dict(contribution),
                }
            )

        current_state = game.get_state()
        trades_proposed = 0
        trades_executed = 0
        allocations: list[dict[str, Any]] = []

        for _ in range(self.max_allocations_per_round):
            candidate_allocations: list[tuple[tuple[int, int, int, int], int, ResourceBundle]] = []

            for agent_state in current_state.agents:
                _, missing = _agent_target_and_missing(game, current_state, agent_state.id)
                if not missing:
                    continue

                bundle = {
                    resource: min(self.pool.get(resource, 0), quantity)
                    for resource, quantity in missing.items()
                    if min(self.pool.get(resource, 0), quantity) > 0
                }
                if not bundle:
                    continue

                policy = game.policies[agent_state.id]
                if not policy.accepts_pool_allocation(
                    agent_id=agent_state.id,
                    bundle=bundle,
                    state=current_state,
                    build_rules=game.build_rules,
                ):
                    continue

                candidate_allocations.append(
                    (
                        self._allocation_priority(agent_state, missing, bundle),
                        agent_state.id,
                        bundle,
                    )
                )

            trades_proposed += len(candidate_allocations)
            if not candidate_allocations:
                break

            candidate_allocations.sort(key=lambda item: item[0])
            _, recipient_id, chosen_bundle = candidate_allocations[0]
            recipient = game.get_agent_by_id(recipient_id)

            for resource, quantity in chosen_bundle.items():
                self.pool[resource] -= quantity
                recipient.stock[resource] += quantity

            trades_executed += 1
            allocations.append(
                {
                    "recipient_id": recipient_id,
                    "bundle": dict(chosen_bundle),
                }
            )
            current_state = game.get_state()

        return InstitutionResult(
            institution_name=self.name,
            trades_proposed=trades_proposed,
            trades_executed=trades_executed,
            details={
                "pool_contributions": contributions,
                "pool_allocations": allocations,
                "remaining_pool": dict(self.pool),
            },
        )


class RoundLocalPublicPoolInstitution(PublicPoolInstitution):
    """Public pool whose unallocated contributions are returned each round.

    The validated :class:`PublicPoolInstitution` retains unused stock across
    rounds. That persistence is useful in the fixed benchmark, but it creates a
    hidden dormant state when a meta-planner switches institutions. This
    planner-specific variant preserves the same contribution and allocation
    logic, then returns every unallocated unit to the contributors before the
    institution phase ends. The inherited benchmark institution is unchanged.
    """

    name = "public_pool_round_local"

    @staticmethod
    def _integer_proportional_returns(
        *,
        remaining: int,
        contributions: list[tuple[int, int]],
    ) -> dict[int, int]:
        """Allocate fungible leftovers proportionally with deterministic ties."""
        if remaining <= 0 or not contributions:
            return {}
        total = sum(quantity for _, quantity in contributions)
        if total <= 0:
            return {}

        exact = [
            (agent_id, remaining * quantity / total)
            for agent_id, quantity in contributions
        ]
        returns = {agent_id: int(value) for agent_id, value in exact}
        assigned = sum(returns.values())
        remainder = remaining - assigned
        if remainder > 0:
            ranked = sorted(
                exact,
                key=lambda item: (-(item[1] - int(item[1])), item[0]),
            )
            for agent_id, _ in ranked[:remainder]:
                returns[agent_id] += 1
        return {agent_id: quantity for agent_id, quantity in returns.items() if quantity > 0}

    def resolve(self, game, state: GameState) -> InstitutionResult:
        result = super().resolve(game=game, state=state)
        contributions = result.details.get("pool_contributions", [])
        returns_by_agent: dict[int, ResourceBundle] = {}

        for resource in RESOURCES:
            remaining = int(self.pool.get(resource, 0))
            if remaining <= 0:
                continue

            contributed = [
                (
                    int(row["agent_id"]),
                    int(row.get("bundle", {}).get(resource, 0)),
                )
                for row in contributions
                if int(row.get("bundle", {}).get(resource, 0)) > 0
            ]
            allocations = self._integer_proportional_returns(
                remaining=remaining,
                contributions=contributed,
            )
            for agent_id, quantity in allocations.items():
                agent = game.get_agent_by_id(agent_id)
                agent.stock[resource] += quantity
                returns_by_agent.setdefault(agent_id, {})[resource] = quantity
                self.pool[resource] -= quantity

        # Defensive conservation check: the round-local pool must never retain
        # hidden stock after the selected institution has finished operating.
        if any(int(quantity) != 0 for quantity in self.pool.values()):
            raise RuntimeError(
                "RoundLocalPublicPoolInstitution failed to return all leftovers."
            )

        result.institution_name = self.name
        result.details["pool_returns"] = [
            {"agent_id": agent_id, "bundle": bundle}
            for agent_id, bundle in sorted(returns_by_agent.items())
        ]
        result.details["remaining_pool"] = dict(self.pool)
        return result
