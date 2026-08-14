"""
Shared resource vocabulary and access-profile definitions.

This module is intentionally low-level: other modules import `RESOURCES`,
`ResourceAccessProfile`, and profile dictionaries from here so the rest of the
simulation can agree on one resource ordering and naming scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


ResourceName = str
Stock = Dict[ResourceName, int]
ProbabilityProfile = Dict[ResourceName, float]


RESOURCES: tuple[ResourceName, ...] = (
    "materials",
    "components",
    "food",
    "energy",
    "knowledge",
)


def empty_stock() -> Stock:
    """Return a fresh stock dictionary containing every resource at zero."""
    return {resource: 0 for resource in RESOURCES}


@dataclass(frozen=True)
class ResourceAccessProfile:
    """
    Defines how likely an agent is to receive each resource.
    """
    name: str
    probabilities: ProbabilityProfile
    primary_resource: ResourceName | None = None
    secondary_resource: ResourceName | None = None

    def normalized_probabilities(self) -> np.ndarray:
        """Return profile weights in fixed resource order, normalized to one."""
        values = np.array([self.probabilities[r] for r in RESOURCES], dtype=float)

        if np.any(values < 0):
            raise ValueError(f"Profile {self.name} contains negative probabilities.")

        total = values.sum()

        if total <= 0:
            raise ValueError(f"Profile {self.name} has no positive probabilities.")

        return values / total


ACCESS_PROFILES: dict[str, ResourceAccessProfile] = {
    "construction": ResourceAccessProfile(
        name="construction",
        probabilities={
            "materials": 0.40,
            "components": 0.35,
            "food": 0.10,
            "energy": 0.10,
            "knowledge": 0.05,
        },
    ),
    "agrarian": ResourceAccessProfile(
        name="agrarian",
        probabilities={
            "materials": 0.15,
            "components": 0.10,
            "food": 0.45,
            "energy": 0.20,
            "knowledge": 0.10,
        },
    ),
    "industrial": ResourceAccessProfile(
        name="industrial",
        probabilities={
            "materials": 0.15,
            "components": 0.25,
            "food": 0.05,
            "energy": 0.35,
            "knowledge": 0.20,
        },
    ),
    "research": ResourceAccessProfile(
        name="research",
        probabilities={
            "materials": 0.05,
            "components": 0.15,
            "food": 0.10,
            "energy": 0.30,
            "knowledge": 0.40,
        },
    ),
    "balanced": ResourceAccessProfile(
        name="balanced",
        probabilities={
            "materials": 0.20,
            "components": 0.20,
            "food": 0.20,
            "energy": 0.20,
            "knowledge": 0.20,
        },
    ),
}


def create_balanced_access_profiles(
    primary_probability: float = 0.45,
    secondary_probability: float = 0.25,
) -> dict[str, ResourceAccessProfile]:
    """
    Generate one neutral asymmetric profile per resource.

    Each profile gives one resource primary weight, the next resource in the
    cycle secondary weight, and spreads the remaining probability evenly over
    the others. Experiment runners use these generated profiles so that every
    resource appears once in each special role.
    """
    if len(RESOURCES) < 3:
        raise ValueError("Balanced access profiles require at least 3 resources.")

    if primary_probability <= 0:
        raise ValueError("primary_probability must be positive.")

    if secondary_probability <= 0:
        raise ValueError("secondary_probability must be positive.")

    if primary_probability + secondary_probability >= 1:
        raise ValueError(
            "primary_probability + secondary_probability must be less than 1."
        )

    remaining_probability = 1 - primary_probability - secondary_probability
    low_probability = remaining_probability / (len(RESOURCES) - 2)

    profiles: dict[str, ResourceAccessProfile] = {}

    for index, primary_resource in enumerate(RESOURCES):
        secondary_resource = RESOURCES[(index + 1) % len(RESOURCES)]
        profile_name = (
            f"balanced_primary_{primary_resource}_secondary_{secondary_resource}"
        )

        probabilities = {
            resource: low_probability
            for resource in RESOURCES
        }
        probabilities[primary_resource] = primary_probability
        probabilities[secondary_resource] = secondary_probability

        profiles[profile_name] = ResourceAccessProfile(
            name=profile_name,
            probabilities=probabilities,
            primary_resource=primary_resource,
            secondary_resource=secondary_resource,
        )

    return profiles


def create_random_balanced_access_profiles(
    rng: np.random.Generator,
    primary_probability: float = 0.45,
    secondary_probability: float = 0.25,
) -> dict[str, ResourceAccessProfile]:
    """
    Generate balanced profiles with randomized primary-secondary pairings.

    Every resource appears exactly once as primary and exactly once as
    secondary. The secondary assignments are sampled as a shuffled derangement
    so no resource is paired with itself.
    """
    if len(RESOURCES) < 3:
        raise ValueError("Balanced access profiles require at least 3 resources.")

    if primary_probability <= 0:
        raise ValueError("primary_probability must be positive.")

    if secondary_probability <= 0:
        raise ValueError("secondary_probability must be positive.")

    if primary_probability + secondary_probability >= 1:
        raise ValueError(
            "primary_probability + secondary_probability must be less than 1."
        )

    primary_resources = list(RESOURCES)
    secondary_resources = list(RESOURCES)

    while True:
        rng.shuffle(secondary_resources)
        if all(
            primary != secondary
            for primary, secondary in zip(primary_resources, secondary_resources)
        ):
            break

    remaining_probability = 1 - primary_probability - secondary_probability
    low_probability = remaining_probability / (len(RESOURCES) - 2)

    profiles: dict[str, ResourceAccessProfile] = {}

    for primary_resource, secondary_resource in zip(primary_resources, secondary_resources):
        profile_name = (
            f"balanced_primary_{primary_resource}_secondary_{secondary_resource}"
        )
        probabilities = {
            resource: low_probability
            for resource in RESOURCES
        }
        probabilities[primary_resource] = primary_probability
        probabilities[secondary_resource] = secondary_probability

        profiles[profile_name] = ResourceAccessProfile(
            name=profile_name,
            probabilities=probabilities,
            primary_resource=primary_resource,
            secondary_resource=secondary_resource,
        )

    return profiles


BALANCED_ACCESS_PROFILES = create_balanced_access_profiles()
