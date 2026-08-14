"""
Immutable public-state views.

Policies and institutions should reason from these snapshots instead of
touching live `Agent` objects directly. That keeps decision logic separate from
mutation and makes the round flow easier to test and extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from engine.resources import Stock


@dataclass(frozen=True)
class PublicAgentState:
    """
    Publicly observable state of one agent.

    This is what other agents / institutions may observe.
    We can later decide whether stock should be fully public or partially hidden.
    """
    id: int
    stock: Stock
    score: int
    infrastructure: int
    production_sites: int
    advanced_sites: int
    innovation: int
    resource_draws: int


@dataclass(frozen=True)
class GameState:
    """
    Snapshot of the full game state at a given round.
    """
    round_number: int
    agents: tuple[PublicAgentState, ...]
    infrastructure_leader_id: int | None = None
    innovation_leader_id: int | None = None
    infrastructure_leader_threshold: int = 5
    innovation_leader_threshold: int = 3
    max_builds_per_agent_per_round: int | None = None

    def agent_state(self, agent_id: int) -> PublicAgentState:
        """Look up one public agent snapshot or raise for an unknown identifier."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent

        raise KeyError(f"No agent with id {agent_id}.")
