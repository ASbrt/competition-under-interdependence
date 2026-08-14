"""
Round orchestration and cross-agent mutation.

`Game` owns the live agents, runs the turn order, applies institution-level
trade execution, applies build decisions through `BuildRules`, updates leader
bonuses, and records the round metrics that experiments later summarize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from engine.actions import (
    BuildAction,
    BuildDecision,
    NoBuildAction,
    ResourceBundle,
    TradeOffer,
)
from engine.agents import Agent
from engine.build_rules import BuildRules
from engine.institutions import Institution, InstitutionResult
from engine.policies import AgentPolicy
from engine.resources import RESOURCES
from engine.state import GameState


@dataclass
class Game:
    """
    Live simulation container.

    Most modules return plans or public views; `Game` is the place where those
    plans become real state changes.
    """
    agents: list[Agent]
    policies: dict[int, AgentPolicy]
    build_rules: BuildRules
    institution: Institution
    rng: np.random.Generator
    production_rng: np.random.Generator | None = None

    infrastructure_leader_id: int | None = None
    innovation_leader_id: int | None = None
    infrastructure_leader_threshold: int = 5
    innovation_leader_threshold: int = 3
    infrastructure_leader_bonus: int = 3
    innovation_leader_bonus: int = 3
    max_builds_per_agent_per_round: int | None = 1

    round_number: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    access_bonus_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize independent random streams while preserving compatibility.

        ``rng`` is the institution/policy random stream used by
        existing callers. Production uses ``production_rng`` so the number of
        random tie-breaks consumed by an institution cannot change later
        exogenous production draws. Runners should pass both streams
        explicitly for matched-condition experiments.
        """
        if self.production_rng is None:
            fallback_seed = int(self.rng.integers(0, np.iinfo(np.int64).max))
            self.production_rng = np.random.default_rng(fallback_seed)

    @property
    def institution_rng(self) -> np.random.Generator:
        """Return the random stream reserved for institutions and tie-breaks."""
        return self.rng

    def get_agent_by_id(self, agent_id: int) -> Agent:
        """Return the mutable live agent with the requested identifier."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent

        raise KeyError(f"No agent with id {agent_id}.")

    def get_state(self) -> GameState:
        """Create the immutable public snapshot supplied to decision layers."""
        return GameState(
            round_number=self.round_number,
            agents=tuple(agent.public_state() for agent in self.agents),
            infrastructure_leader_id=self.infrastructure_leader_id,
            innovation_leader_id=self.innovation_leader_id,
            infrastructure_leader_threshold=self.infrastructure_leader_threshold,
            innovation_leader_threshold=self.innovation_leader_threshold,
            max_builds_per_agent_per_round=self.max_builds_per_agent_per_round,
        )

    def can_pay_bundle(self, agent_id: int, bundle: ResourceBundle) -> bool:
        """Check that a bundle is positive and affordable without mutating stock."""
        agent = self.get_agent_by_id(agent_id)

        for resource, quantity in bundle.items():
            if quantity <= 0:
                raise ValueError(
                    f"Bundle quantities must be positive. Got {resource}={quantity}."
                )

            if agent.stock.get(resource, 0) < quantity:
                return False

        return True

    def transfer_bundle(
        self,
        from_agent_id: int,
        to_agent_id: int,
        bundle: ResourceBundle,
    ) -> bool:
        """Move a one-way resource bundle if the source can still pay it."""
        if not self.can_pay_bundle(from_agent_id, bundle):
            return False

        from_agent = self.get_agent_by_id(from_agent_id)
        to_agent = self.get_agent_by_id(to_agent_id)

        for resource, quantity in bundle.items():
            from_agent.stock[resource] -= quantity
            to_agent.stock[resource] += quantity

        return True

    def execute_trade_offer(self, offer: TradeOffer) -> bool:
        """Atomically execute both sides of a feasible reciprocal trade offer."""
        # Check both obligations before changing either stock. Offers may have
        # become infeasible since construction if an earlier trade used stock.
        if not self.can_pay_bundle(offer.responder_id, offer.offered_bundle):
            return False

        if not self.can_pay_bundle(offer.requester_id, offer.requested_bundle):
            return False

        responder = self.get_agent_by_id(offer.responder_id)
        requester = self.get_agent_by_id(offer.requester_id)

        for resource, quantity in offer.offered_bundle.items():
            responder.stock[resource] -= quantity
            requester.stock[resource] += quantity

        for resource, quantity in offer.requested_bundle.items():
            requester.stock[resource] -= quantity
            responder.stock[resource] += quantity

        return True

    def produce_resources(self) -> dict[int, dict[str, int]]:
        """Run one production draw for every agent and return realized bundles."""
        produced_by_agent = {}

        if self.production_rng is None:
            raise RuntimeError("production_rng was not initialized.")

        for agent in self.agents:
            produced_by_agent[agent.id] = agent.produce_resources(self.production_rng)

        return produced_by_agent

    def collect_build_decisions(self, state: GameState) -> list[BuildDecision]:
        """Collect one proposed build per agent for a supplied snapshot.

        The live round loop uses :meth:`apply_policy_builds`, which refreshes
        state between repeated attempts. This helper instead exposes a
        simultaneous one-decision view for diagnostics or alternative runners.
        """
        decisions = []

        for agent in self.agents:
            policy = self.policies[agent.id]

            decision = policy.choose_build_action(
                agent_id=agent.id,
                state=state,
                build_rules=self.build_rules,
            )

            decisions.append(decision)

        return decisions

    def _record_access_bonus_event(
        self,
        agent: Agent,
        decision: BuildAction,
        bonus_resource: str | None,
        old_weight: float | None,
    ) -> None:
        """Append a production-site access change to the event audit trail."""
        if bonus_resource is None or old_weight is None:
            return

        new_weight = agent.effective_access_weights()[bonus_resource]
        self.access_bonus_events.append(
            {
                "round": self.round_number,
                "agent_id": agent.id,
                "build_name": decision.build_name,
                "bonus_resource": bonus_resource,
                "bonus_amount": new_weight - old_weight,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "score_after": agent.score,
                "production_sites_after": agent.production_sites,
                "advanced_sites_after": agent.advanced_sites,
            }
        )

    def _leader_count(self, agent: Agent, kind: str) -> int:
        """Read the build counter associated with one transferable crown."""
        if kind == "infrastructure":
            return agent.infrastructure

        if kind == "innovation":
            return agent.innovation

        raise ValueError(f"Unknown leader kind '{kind}'.")

    def _update_single_leader_bonus(
        self,
        kind: str,
        current_leader_attr: str,
        threshold: int,
        bonus: int,
    ) -> None:
        """Claim or transfer one crown while preserving the incumbent on ties."""
        current_leader_id = getattr(self, current_leader_attr)
        counts_by_agent = {
            agent.id: self._leader_count(agent, kind)
            for agent in self.agents
        }
        highest_count = max(counts_by_agent.values())
        highest_agent_ids = [
            agent_id
            for agent_id, count in counts_by_agent.items()
            if count == highest_count
        ]

        if current_leader_id is None:
            if highest_count >= threshold and len(highest_agent_ids) == 1:
                new_leader = self.get_agent_by_id(highest_agent_ids[0])
                new_leader.score += bonus
                setattr(self, current_leader_attr, new_leader.id)

            return

        current_leader_count = counts_by_agent[current_leader_id]
        surpassing_agent_ids = [
            agent_id
            for agent_id, count in counts_by_agent.items()
            if agent_id != current_leader_id and count > current_leader_count
        ]

        if not surpassing_agent_ids:
            return

        highest_surpassing_count = max(
            counts_by_agent[agent_id]
            for agent_id in surpassing_agent_ids
        )
        next_leader_ids = [
            agent_id
            for agent_id in surpassing_agent_ids
            if counts_by_agent[agent_id] == highest_surpassing_count
        ]

        if len(next_leader_ids) != 1:
            return

        old_leader = self.get_agent_by_id(current_leader_id)
        new_leader = self.get_agent_by_id(next_leader_ids[0])

        old_leader.score -= bonus
        new_leader.score += bonus
        setattr(self, current_leader_attr, new_leader.id)

    def update_leader_bonuses(self) -> None:
        """
        Recompute both Catan-like leader bonuses after a successful build.
        """
        self._update_single_leader_bonus(
            kind="infrastructure",
            current_leader_attr="infrastructure_leader_id",
            threshold=self.infrastructure_leader_threshold,
            bonus=self.infrastructure_leader_bonus,
        )
        self._update_single_leader_bonus(
            kind="innovation",
            current_leader_attr="innovation_leader_id",
            threshold=self.innovation_leader_threshold,
            bonus=self.innovation_leader_bonus,
        )

    def apply_policy_builds(self) -> int:
        """Apply sequential policy-selected builds against refreshed state.

        ``None`` removes the configured gameplay limit, but a defensive ceiling
        of 20 attempts prevents a faulty policy from creating an infinite loop.
        """
        if self.max_builds_per_agent_per_round is not None:
            if self.max_builds_per_agent_per_round <= 0:
                raise ValueError("max_builds_per_agent_per_round must be positive or None.")
            per_agent_attempt_limit = self.max_builds_per_agent_per_round
        else:
            per_agent_attempt_limit = 20

        builds_applied = 0

        for agent in self.agents:
            policy = self.policies[agent.id]

            for _ in range(per_agent_attempt_limit):
                current_state = self.get_state()
                decision = policy.choose_build_action(
                    agent_id=agent.id,
                    state=current_state,
                    build_rules=self.build_rules,
                )

                if isinstance(decision, NoBuildAction):
                    break

                if not isinstance(decision, BuildAction):
                    break

                bonus_resource = None
                old_weight = None

                # Re-check validity during execution.
                # Policies can propose actions, but rules decide whether they are legal.
                if not self.build_rules.can_build(agent, decision.build_name):
                    break

                if decision.build_name == "production_site":
                    bonus_resource = self.build_rules.choose_production_site_bonus_resource(
                        agent
                    )
                    if bonus_resource is not None:
                        old_weight = agent.effective_access_weights()[bonus_resource]

                self.build_rules.apply_build(agent, decision.build_name)
                self._record_access_bonus_event(
                    agent=agent,
                    decision=decision,
                    bonus_resource=bonus_resource,
                    old_weight=old_weight,
                )

                # Leader bonuses depend on the post-build counts, so update
                # immediately after every successful build.
                self.update_leader_bonuses()
                builds_applied += 1

        return builds_applied

    def compute_metrics(
        self,
        produced: dict[int, dict[str, int]],
        institution_result: InstitutionResult,
        builds_applied: int,
    ) -> dict[str, Any]:
        """Summarize post-build state for history and later analysis.

        Institution event counts retain institution-specific semantics: one
        execution can be a barter trade, subsidy, or pooled allocation.
        """
        scores = [agent.score for agent in self.agents]
        min_score = min(scores)
        max_score = max(scores)

        total_resources = sum(
            sum(agent.stock.values())
            for agent in self.agents
        )

        total_by_resource = {
            resource: sum(agent.stock.get(resource, 0) for agent in self.agents)
            for resource in RESOURCES
        }

        return {
            "round": self.round_number,
            "institution": institution_result.institution_name,
            "total_score": sum(scores),
            "min_score": min_score,
            "max_score": max_score,
            "score_gap": max_score - min_score,
            "infrastructure_leader_id": self.infrastructure_leader_id,
            "innovation_leader_id": self.innovation_leader_id,
            "infrastructure_leader_bonus_active": self.infrastructure_leader_id is not None,
            "innovation_leader_bonus_active": self.innovation_leader_id is not None,
            "total_resources": total_resources,
            "builds_applied": builds_applied,
            "trades_proposed": institution_result.trades_proposed,
            "trades_executed": institution_result.trades_executed,
            "total_infrastructure": sum(agent.infrastructure for agent in self.agents),
            "idle_infrastructure": sum(
                max(
                    0,
                    agent.infrastructure
                    - 2 * (agent.production_sites + agent.advanced_sites),
                )
                for agent in self.agents
            ),
            "total_production_sites": sum(agent.production_sites for agent in self.agents),
            "total_advanced_sites": sum(agent.advanced_sites for agent in self.agents),
            "total_innovation": sum(agent.innovation for agent in self.agents),
            "total_materials": total_by_resource["materials"],
            "total_components": total_by_resource["components"],
            "total_food": total_by_resource["food"],
            "total_energy": total_by_resource["energy"],
            "total_knowledge": total_by_resource["knowledge"],
        }

    def step(self) -> dict[str, Any]:
        """
        Generic round structure.

        The institution can be NoTradeInstitution, BilateralTradeInstitution,
        CentralMarketInstitution, etc.
        """
        produced = self.produce_resources()

        # Institutions see the post-production state and may change stocks
        # through trade before build choices are collected.
        state_after_production = self.get_state()

        institution_result = self.institution.resolve(
            game=self,
            state=state_after_production,
        )

        # Build choices are always made against the latest post-institution
        # state, and with max_builds_per_agent_per_round > 1 each agent gets a
        # fresh public state before every additional build attempt.
        builds_applied = self.apply_policy_builds()

        metrics = self.compute_metrics(
            produced=produced,
            institution_result=institution_result,
            builds_applied=builds_applied,
        )

        self.history.append(metrics)

        self.round_number += 1

        return metrics
