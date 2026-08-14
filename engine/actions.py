"""
Small immutable data objects shared across the engine.

Policies use these objects to describe what they want to do, institutions use
them to coordinate bargaining, and `Game` consumes the concrete actions during
execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from engine.build_rules import BuildName
from engine.resources import ResourceName


ResourceBundle: TypeAlias = dict[ResourceName, int]


@dataclass(frozen=True)
class BuildAction:
    """
    Concrete build action chosen by an agent.
    """
    agent_id: int
    build_name: BuildName


@dataclass(frozen=True)
class NoBuildAction:
    """
    Explicit choice to build nothing this round.

    Useful when we want all policies to return an action object
    instead of returning None.
    """
    agent_id: int


BuildDecision: TypeAlias = BuildAction | NoBuildAction


@dataclass(frozen=True)
class BuildTarget:
    """
    Build project an agent is currently aiming toward.
    """
    agent_id: int
    build_name: BuildName


@dataclass(frozen=True)
class TradeRequest:
    """
    General request for a missing resource.
    """
    requester_id: int
    requested_resource: ResourceName
    quantity: int = 1


@dataclass(frozen=True)
class TradeOffer:
    """
    Concrete bilateral offer in response to a request.
    """
    responder_id: int
    requester_id: int
    offered_bundle: ResourceBundle
    requested_bundle: ResourceBundle


@dataclass(frozen=True)
class TradeIntention:
    """
    Compatibility shim for trade-oriented imports.
    """
    agent_id: int
    offer_resource: ResourceName
    request_resource: ResourceName
    quantity: int = 1


@dataclass(frozen=True)
class TradeProposal:
    """
    Compatibility shim for trade-oriented imports.
    """
    proposer_id: int
    receiver_id: int
    proposer_resource: ResourceName
    receiver_resource: ResourceName
    quantity: int = 1
