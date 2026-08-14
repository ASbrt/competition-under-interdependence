"""Baseline institutional policies for paired online-planner evaluation.

The key question is not only whether the Q-policy uses good institutions, but
whether it uses them at useful times. The frequency-informed and shuffled
baselines keep much of the learned action preference while weakening the timing
signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from experiments.adaptive.capacity_coordination import CapacityObservation
from experiments.adaptive.online_q_planner import ACTION_ORDER, OnlineTabularQPlanner


class PlannerPolicy(Protocol):
    name: str
    label: str

    def reset_game(self, *, seed: int, scenario: str) -> None:
        ...

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        ...


def ordered_available(available_actions: Sequence[str]) -> list[str]:
    return [action for action in ACTION_ORDER if action in available_actions]


@dataclass
class LearnedQPolicy:
    planner: OnlineTabularQPlanner
    name: str = "learned_q"
    label: str = "Learned Q policy"

    def reset_game(self, *, seed: int, scenario: str) -> None:
        return None

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        return self.planner.choose_action(
            observation=observation,
            available_actions=available_actions,
            rng=rng,
            epsilon=0.0,
        )


@dataclass
class PermanentActionPolicy:
    action: str = "bilateral_3pass"
    name: str = "permanent_bilateral_3pass"
    label: str = "Permanent bilateral 3-pass"

    def reset_game(self, *, seed: int, scenario: str) -> None:
        return None

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        if self.action in available_actions:
            return self.action
        return "bilateral_3pass"


@dataclass
class UniformRandomFeasiblePolicy:
    name: str = "random_feasible"
    label: str = "Uniform random feasible"

    def reset_game(self, *, seed: int, scenario: str) -> None:
        return None

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        available = ordered_available(available_actions)
        return str(rng.choice(available))


@dataclass
class SimpleHeuristicPolicy:
    name: str = "simple_state_heuristic"
    label: str = "Simple state heuristic"

    def reset_game(self, *, seed: int, scenario: str) -> None:
        return None

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        available = set(available_actions)
        if observation.score_gap >= 14 and "subsidized_catch_up" in available:
            return "subsidized_catch_up"
        if observation.voluntary_match_rate < 0.34:
            for action in ("central_full", "central_cap2", "clearinghouse"):
                if action in available:
                    return action
        if observation.total_missing_units >= 10:
            for action in ("public_pool", "clearinghouse"):
                if action in available:
                    return action
        if observation.coordination_capacity >= 7 and "central_full" in available:
            return "central_full"
        return "bilateral_3pass"


@dataclass
class FrequencyMatchedRandomPolicy:
    """Random feasible policy weighted by learned aggregate action counts.

    The weights are renormalized over the currently feasible actions, so the
    realized action shares need not exactly match the learned policy.
    """
    action_counts: Mapping[str, int]
    name: str = "frequency_informed_random"
    label: str = "Frequency-informed random feasible"

    def reset_game(self, *, seed: int, scenario: str) -> None:
        return None

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        available = ordered_available(available_actions)
        weights = np.asarray(
            [max(0, int(self.action_counts.get(action, 0))) for action in available],
            dtype=np.float64,
        )
        if weights.sum() <= 0:
            weights = np.ones(len(available), dtype=np.float64)
        weights = weights / weights.sum()
        return str(rng.choice(available, p=weights))


class ShuffledLearnedTimingPolicy:
    """Reuse learned per-game actions after shuffling their order.

    This tests whether the learned sequence helped because of state-dependent
    timing or merely because it used a reasonable set of institutions in that
    game.
    """
    name = "shuffled_learned_sequence"
    label = "Shuffled learned sequence"

    def __init__(
        self,
        *,
        action_sequences: Mapping[tuple[int, str], Sequence[str]],
        fallback_counts: Mapping[str, int],
    ):
        self.action_sequences = {
            key: tuple(actions) for key, actions in action_sequences.items()
        }
        self.fallback = FrequencyMatchedRandomPolicy(fallback_counts)
        self._queue: deque[str] = deque()
        self._shuffled = False

    def reset_game(self, *, seed: int, scenario: str) -> None:
        self._queue = deque(self.action_sequences.get((seed, scenario), ()))
        self._shuffled = False

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
    ) -> str:
        if self._queue and not self._shuffled:
            shuffled = list(self._queue)
            rng.shuffle(shuffled)
            self._queue = deque(shuffled)
            self._shuffled = True
        for _ in range(len(self._queue)):
            action = self._queue.popleft()
            if action in available_actions:
                return action
        return self.fallback.choose_action(observation, available_actions, rng)
