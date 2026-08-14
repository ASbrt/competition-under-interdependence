"""
Build definitions and build-side rule logic.

`BuildRules` is the layer that knows what can be built, what it costs, what
structural prerequisites apply, and how a successful build mutates a real live
agent. Policies can query it from public state, while only `Game` should use it
to mutate real agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from engine.agents import Agent
from engine.resources import RESOURCES, ResourceName, Stock
from engine.state import PublicAgentState


BuildName = str
Cost = Dict[str, int]
AgentLike = Agent | PublicAgentState


@dataclass(frozen=True)
class BuildProject:
    """
    A buildable project in the game.

    cost:
        Resource bundle required to build it.

    points:
        Immediate score gained when the project is built.

    description:
        Human-readable explanation for debugging / documentation.
    """
    name: BuildName
    cost: Cost
    points: int
    description: str


DEFAULT_BUILD_PROJECTS: dict[BuildName, BuildProject] = {
    "infrastructure": BuildProject(
        name="infrastructure",
        cost={
            "materials": 1,
            "components": 1,
        },
        points=0,
        description="Expansion capacity. Two infrastructure units support one site.",
    ),
    "production_site": BuildProject(
        name="production_site",
        cost={
            "materials": 1,
            "components": 1,
            "food": 1,
            "energy": 1,
        },
        points=3,
        description="Productive site. Requires free infrastructure capacity and adds resource production.",
    ),
    "advanced_site": BuildProject(
        name="advanced_site",
        cost={
            "energy": 2,
            "knowledge": 3,
        },
        points=5,
        description="Upgrade of an existing production site into a more productive advanced site.",
    ),
    "innovation": BuildProject(
        name="innovation",
        cost={
            "food": 1,
            "energy": 1,
            "knowledge": 1,
        },
        points=1,
        description="Non-productive scoring path. Counts toward innovation leadership.",
    ),
}


class BuildRules:
    """
    Rule layer for build actions.

    This class knows:
    - what projects exist
    - what they cost
    - what prerequisites they have
    - how they modify an agent when built

    Agents should not directly implement these rules.
    """

    def __init__(
        self,
        projects: dict[BuildName, BuildProject] | None = None,
        max_scoring_innovation: int = 3,
        enable_access_expansion: bool = True,
        production_site_access_bonus: float = 0.10,
    ):
        """Configure build definitions, scoring caps, and access expansion."""
        self.projects = projects if projects is not None else DEFAULT_BUILD_PROJECTS
        if max_scoring_innovation < 0:
            raise ValueError("max_scoring_innovation must be non-negative.")
        if production_site_access_bonus < 0:
            raise ValueError("production_site_access_bonus must be non-negative.")

        self.max_scoring_innovation = max_scoring_innovation
        self.enable_access_expansion = enable_access_expansion
        self.production_site_access_bonus = production_site_access_bonus

    def get_project(self, build_name: BuildName) -> BuildProject:
        """Return a build definition with a useful error for unknown names."""
        if build_name not in self.projects:
            known = ", ".join(self.projects.keys())
            raise KeyError(f"Unknown build project '{build_name}'. Known projects: {known}")

        return self.projects[build_name]

    def can_afford(self, stock: Stock, cost: Cost) -> bool:
        """
        Check whether a resource stock contains the required resources.
        """
        for resource, amount in cost.items():
            if stock.get(resource, 0) < amount:
                return False

        return True

    def missing_resources(self, stock: Stock, cost: Cost) -> Stock:
        """
        Return the missing resources for a cost.

        Example:
            stock = {"materials": 1, "components": 0}
            cost  = {"materials": 1, "components": 1}

            returns {"components": 1}
        """
        missing: Stock = {}

        for resource, amount in cost.items():
            available = stock.get(resource, 0)
            gap = max(0, amount - available)

            if gap > 0:
                missing[resource] = gap

        return missing

    def missing_resources_for_build(self, agent: AgentLike, build_name: BuildName) -> Stock:
        """
        Return missing resources for a specific build project.

        This only checks resources. It does not check structural prerequisites.
        """
        project = self.get_project(build_name)
        return self.missing_resources(agent.stock, project.cost)

    def free_site_capacity(self, agent: AgentLike) -> int:
        """
        Return unused site capacity from publicly visible fields.
        """
        site_capacity = agent.infrastructure // 2
        used_site_capacity = agent.production_sites + agent.advanced_sites
        return site_capacity - used_site_capacity

    def has_structural_prerequisites(
        self,
        agent: AgentLike,
        build_name: BuildName,
    ) -> bool:
        """
        Check non-resource prerequisites.

        Examples:
        - production sites require free site capacity
        - advanced sites require an existing production site
        """
        if build_name == "infrastructure":
            return True

        if build_name == "production_site":
            return self.free_site_capacity(agent) > 0

        if build_name == "advanced_site":
            return agent.production_sites > 0

        if build_name == "innovation":
            return True

        self.get_project(build_name)  # raises useful error if unknown
        return False

    def can_build(self, agent: AgentLike, build_name: BuildName) -> bool:
        """
        Check whether an agent can build this project now.
        """
        project = self.get_project(build_name)

        has_resources = self.can_afford(agent.stock, project.cost)
        has_structure = self.has_structural_prerequisites(agent, build_name)

        return has_resources and has_structure

    def pay_cost(self, stock: Stock, cost: Cost) -> None:
        """
        Remove resources from stock in-place.

        Assumes affordability has already been checked.
        """
        for resource, amount in cost.items():
            stock[resource] -= amount

    def choose_production_site_bonus_resource(
        self,
        agent: Agent,
    ) -> ResourceName | None:
        """
        Pick the weakest current access weight that should receive the next
        production-site access bonus.

        This stays as a small helper so build mutation and experiment logging
        can rely on the same choice rule.
        """
        if not self.enable_access_expansion:
            return None

        if self.production_site_access_bonus <= 0:
            return None

        effective_weights = agent.effective_access_weights()
        return min(
            RESOURCES,
            key=lambda resource: (
                effective_weights[resource],
                RESOURCES.index(resource),
            ),
        )

    def apply_build(self, agent: Agent, build_name: BuildName) -> None:
        """
        Apply a build action to an agent.

        This mutates the agent:
        - pays resources
        - increases score
        - changes built assets
        """
        if not self.can_build(agent, build_name):
            raise ValueError(f"Agent {agent.id} cannot build '{build_name}'.")

        project = self.get_project(build_name)

        self.pay_cost(agent.stock, project.cost)

        # Innovation is special: it can continue growing after its direct-score
        # cap because later systems still care about the total innovation count.
        if build_name == "infrastructure":
            agent.score += project.points
            agent.infrastructure += 1

        elif build_name == "production_site":
            agent.score += project.points
            agent.production_sites += 1
            weakest_resource = self.choose_production_site_bonus_resource(agent)
            if weakest_resource is not None:
                agent.access_bonuses[weakest_resource] += self.production_site_access_bonus

        elif build_name == "advanced_site":
            agent.score += project.points
            agent.production_sites -= 1
            agent.advanced_sites += 1

        elif build_name == "innovation":
            agent.innovation += 1
            if agent.innovation <= self.max_scoring_innovation:
                agent.score += project.points

    def available_builds(self, agent: AgentLike) -> list[BuildName]:
        """
        Return all build projects the agent can currently build.
        """
        return [
            build_name
            for build_name in self.projects
            if self.can_build(agent, build_name)
        ]
