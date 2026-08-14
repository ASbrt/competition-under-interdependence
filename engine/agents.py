"""
Mutable agent model.

`Agent` stores the live private state that only `Game` and `BuildRules` should
mutate directly. Other modules typically work through `public_state()` so that
policies and institutions can reason from a read-only snapshot.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from engine.resources import (
    RESOURCES,
    ResourceName,
    Stock,
    ResourceAccessProfile,
    empty_stock,
)
from engine.state import PublicAgentState


def empty_access_bonuses() -> dict[ResourceName, float]:
    """Return a fresh zero-valued access-bonus mapping for one agent."""
    return {resource: 0.0 for resource in RESOURCES}


@dataclass
class Agent:
    """
    Live in-game agent with private stock and build counters.
    """
    id: int
    access_profile: ResourceAccessProfile
    stock: Stock = field(default_factory=empty_stock)
    access_bonuses: dict[ResourceName, float] = field(default_factory=empty_access_bonuses)

    score: int = 0

    infrastructure: int = 0
    production_sites: int = 0
    advanced_sites: int = 0
    innovation: int = 0

    base_draws: int = 2
    max_draws: int = 7

    def current_resource_draws(self) -> int:
        """
        Resource production capacity.

        Production sites add +1 draw.
        Advanced sites add +2 draws.
        """
        draws = (
            self.base_draws
            + self.production_sites
            + 2 * self.advanced_sites
        )

        return min(draws, self.max_draws)

    def produce_resources(self, rng: np.random.Generator) -> Stock:
        """
        Draw resources according to this agent's access profile and add them
        directly to the agent's personal stock.
        """
        n_draws = self.current_resource_draws()
        effective_weights = self.effective_access_weights()
        total_weight = sum(effective_weights.values())
        probabilities = np.array(
            [effective_weights[resource] / total_weight for resource in RESOURCES],
            dtype=float,
        )

        draws = rng.multinomial(n=n_draws, pvals=probabilities)

        produced = {
            resource: int(amount)
            for resource, amount in zip(RESOURCES, draws)
        }

        for resource, amount in produced.items():
            self.stock[resource] += amount

        return produced

    def effective_access_weights(self) -> dict[ResourceName, float]:
        """
        Combine the base access profile with any dynamic access bonuses.
        """
        return {
            resource: self.access_profile.probabilities[resource] + self.access_bonuses[resource]
            for resource in RESOURCES
        }

    def site_capacity(self) -> int:
        """
        Each two infrastructure units support one production site.
        """
        return self.infrastructure // 2

    def used_site_capacity(self) -> int:
        """
        Production sites and advanced sites both occupy one site slot.
        """
        return self.production_sites + self.advanced_sites

    def free_site_capacity(self) -> int:
        """
        Calculate free site capacity
        """
        return self.site_capacity() - self.used_site_capacity()

    def public_state(self) -> PublicAgentState:
        """
        Convert live mutable state into the public snapshot seen by policies.
        """
        return PublicAgentState(
            id=self.id,
            stock=self.stock.copy(),
            score=self.score,
            infrastructure=self.infrastructure,
            production_sites=self.production_sites,
            advanced_sites=self.advanced_sites,
            innovation=self.innovation,
            resource_draws=self.current_resource_draws(),
        )
