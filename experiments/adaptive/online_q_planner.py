"""Online tabular Q-learning planner for institutional selection.

Unlike the former offline learner, this planner updates from transitions it
creates while interacting with training games. The learned table is frozen only
for held-out evaluation, which is standard reinforcement-learning practice.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.adaptive.capacity_coordination import (
    DEFAULT_ACTION_SPECS,
    CapacityObservation,
)


ACTION_ORDER: tuple[str, ...] = (
    "bilateral_3pass",
    "clearinghouse",
    "public_pool",
    "subsidized_catch_up",
    "central_cap2",
    "equity_cap2",
    "central_full",
)

PREVIOUS_ACTION_CODE = {
    action: index for index, action in enumerate(ACTION_ORDER)
}
StateKey = tuple[int, ...]
Transition = dict[str, object]


def _bin(value: float, thresholds: Sequence[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return index
    return len(thresholds)


def discretize_observation(observation: CapacityObservation) -> StateKey:
    """Map the observed economy to seven interpretable categorical features."""
    phase = min(
        3,
        int(4 * observation.round_number / observation.total_rounds),
    )

    capacity_ratio = (
        observation.coordination_capacity
        / max(1, observation.max_coordination_capacity)
    )
    if observation.coordination_capacity == 0:
        capacity_bin = 0
    elif capacity_ratio <= 1 / 3:
        capacity_bin = 1
    elif capacity_ratio <= 2 / 3:
        capacity_bin = 2
    else:
        capacity_bin = 3

    if observation.total_missing_units >= 11 or observation.blocked_agents >= 5:
        bottleneck_bin = 2
    elif observation.total_missing_units >= 8 or observation.blocked_agents >= 3:
        bottleneck_bin = 1
    else:
        bottleneck_bin = 0

    if observation.current_requests == 0:
        viability_bin = 0
    elif observation.voluntary_match_rate < 0.34:
        viability_bin = 1
    elif observation.voluntary_match_rate < 0.67:
        viability_bin = 2
    else:
        viability_bin = 3

    if observation.score_gap < 7:
        inclusion_bin = 0
    elif observation.score_gap < 14:
        inclusion_bin = 1
    else:
        inclusion_bin = 2

    if observation.idle_infrastructure < 5:
        idle_bin = 0
    elif observation.idle_infrastructure < 10:
        idle_bin = 1
    else:
        idle_bin = 2

    previous_action_bin = PREVIOUS_ACTION_CODE.get(
        observation.previous_action,
        PREVIOUS_ACTION_CODE["bilateral_3pass"],
    )

    return (
        phase,
        capacity_bin,
        bottleneck_bin,
        viability_bin,
        inclusion_bin,
        idle_bin,
        previous_action_bin,
    )


STATE_V2_FEATURES: tuple[dict[str, object], ...] = (
    {
        "name": "rounds_remaining_bin",
        "thresholds": (6, 13),
        "description": "Remaining horizon; branch sample quartiles were about 6, 12.5, and 19 rounds.",
    },
    {
        "name": "capacity_affordability_bin",
        "thresholds": (3, 5),
        "description": "Coordination capacity level as a compact proxy for affordable institution families.",
    },
    {
        "name": "score_gap_bin",
        "thresholds": (5, 12),
        "description": "Inequality gap; branch sample median was 5 and upper quartile was about 12.",
    },
    {
        "name": "bottom_two_score_bin",
        "thresholds": (1, 3),
        "description": "Absolute lower-tail development level using the bottom-two mean score.",
    },
    {
        "name": "agents_one_unit_short_bin",
        "thresholds": (1, 2),
        "description": "Count of agents exactly one resource unit short of their current build target.",
    },
    {
        "name": "blocked_agents_bin",
        "thresholds": (2, 4),
        "description": "Count of agents blocked from their current build target.",
    },
    {
        "name": "request_count_bin",
        "thresholds": (2, 4),
        "description": "Number of current voluntary trade requests.",
    },
    {
        "name": "acceptable_offer_count_bin",
        "thresholds": (0, 2),
        "description": "Number of feasible acceptable voluntary offers in the current post-production state.",
    },
    {
        "name": "previous_action_bin",
        "thresholds": (),
        "description": "Previous institution, preserving switching-cost context.",
    },
)


def state_v2_theoretical_size() -> int:
    size = len(ACTION_ORDER)
    for feature in STATE_V2_FEATURES:
        if feature["name"] == "previous_action_bin":
            continue
        size *= len(feature["thresholds"]) + 1
    return size


STATE_V3_FEATURES: tuple[dict[str, object], ...] = (
    {
        "name": "rounds_remaining_bin",
        "levels": 4,
        "thresholds": (5, 10, 15),
        "description": "Four horizon bins across the 20-round episode.",
    },
    {
        "name": "affordability_tier",
        "levels": 4,
        "thresholds": (),
        "description": "Highest currently feasible institutional family: bilateral only, low-cost coordinated, capped central, or full central.",
    },
    {
        "name": "lower_tail_pressure",
        "levels": 3,
        "thresholds": (),
        "description": "Composite pressure from bottom-two score level and mean-minus-bottom-two gap.",
    },
    {
        "name": "build_proximity",
        "levels": 3,
        "thresholds": (0, 1),
        "description": "Agents exactly one resource unit short: 0, 1, or 2+.",
    },
    {
        "name": "market_thickness",
        "levels": 3,
        "thresholds": (0, 2),
        "description": "Current voluntary requests: 0, 1-2, or 3+.",
    },
    {
        "name": "feasible_voluntary_trade_volume",
        "levels": 3,
        "thresholds": (0, 2),
        "description": "Behaviorally acceptable and payable voluntary exchange units: 0, 1-2, or 3+.",
    },
    {
        "name": "recent_voluntary_viability",
        "levels": 3,
        "thresholds": (0.34, 0.67),
        "description": "Rolling current/probed three-round match-rate viability: low, medium, high.",
    },
    {
        "name": "shortage_concentration",
        "levels": 3,
        "thresholds": (0.40, 0.67),
        "description": "Largest missing-resource share among current build shortages: diffuse, moderate, concentrated.",
    },
    {
        "name": "previous_action_bin",
        "levels": len(ACTION_ORDER),
        "thresholds": (),
        "description": "Previous institution, preserving switching-cost context.",
    },
)


def state_v3_theoretical_size() -> int:
    size = 1
    for feature in STATE_V3_FEATURES:
        size *= int(feature["levels"])
    return size


ACTION_SPEC_BY_NAME = {spec.name: spec for spec in DEFAULT_ACTION_SPECS}


def state_v3_affordability_tier(observation: CapacityObservation) -> int:
    feasible = {
        name
        for name, spec in ACTION_SPEC_BY_NAME.items()
        if name == "bilateral_3pass"
        or spec.commitment_cost(observation.previous_action)
        <= observation.coordination_capacity
    }
    if "central_full" in feasible:
        return 3
    if {"central_cap2", "equity_cap2"} & feasible:
        return 2
    if {"clearinghouse", "public_pool", "subsidized_catch_up"} & feasible:
        return 1
    return 0


def state_v3_lower_tail_pressure(observation: CapacityObservation) -> int:
    lower_tail_gap = observation.mean_score - observation.bottom_two_mean_score
    if observation.bottom_two_mean_score <= 1 or lower_tail_gap >= 6:
        return 2
    if observation.bottom_two_mean_score <= 3 or lower_tail_gap >= 3:
        return 1
    return 0


def state_v3_build_proximity(observation: CapacityObservation) -> int:
    return min(2, int(observation.agents_one_unit_short))


def state_v3_market_thickness(observation: CapacityObservation) -> int:
    return _bin(observation.current_requests, (0, 2))


def state_v3_feasible_volume(observation: CapacityObservation) -> int:
    return _bin(observation.feasible_voluntary_trade_units, (0, 2))


def state_v3_recent_viability(observation: CapacityObservation) -> int:
    return _bin(observation.recent_voluntary_match_rate, (0.34, 0.67))


def state_v3_shortage_concentration(observation: CapacityObservation) -> int:
    return _bin(observation.shortage_concentration, (0.40, 0.67))


def discretize_observation_v3(observation: CapacityObservation) -> StateKey:
    previous_action_bin = PREVIOUS_ACTION_CODE.get(
        observation.previous_action,
        PREVIOUS_ACTION_CODE["bilateral_3pass"],
    )
    return (
        _bin(observation.rounds_remaining, (5, 10, 15)),
        state_v3_affordability_tier(observation),
        state_v3_lower_tail_pressure(observation),
        state_v3_build_proximity(observation),
        state_v3_market_thickness(observation),
        state_v3_feasible_volume(observation),
        state_v3_recent_viability(observation),
        state_v3_shortage_concentration(observation),
        previous_action_bin,
    )


def discretize_observation_v2(observation: CapacityObservation) -> StateKey:
    """Compact state-v2 encoder targeted at branch-diagnostic aliasing.

    This is an alternative state representation used only for supplementary learner checks.
    """
    previous_action_bin = PREVIOUS_ACTION_CODE.get(
        observation.previous_action,
        PREVIOUS_ACTION_CODE["bilateral_3pass"],
    )
    return (
        _bin(observation.rounds_remaining, (6, 13)),
        _bin(observation.coordination_capacity, (3, 5)),
        _bin(observation.score_gap, (5, 12)),
        _bin(observation.bottom_two_mean_score, (1, 3)),
        _bin(observation.agents_one_unit_short, (1, 2)),
        _bin(observation.blocked_agents, (2, 4)),
        _bin(observation.current_requests, (2, 4)),
        _bin(observation.voluntary_acceptable_offers, (0, 2)),
        previous_action_bin,
    )


def state_feature_spec(state_version: str) -> tuple[dict[str, object], ...]:
    if state_version == "v1":
        return (
            {"name": "phase", "description": "Round-number quartile."},
            {"name": "capacity_bin", "description": "Coordination-capacity ratio bin."},
            {"name": "bottleneck_bin", "description": "Missing-units/blocked-agent bin."},
            {"name": "viability_bin", "description": "Voluntary exchange match-rate bin."},
            {"name": "inclusion_bin", "description": "Score-gap bin."},
            {"name": "idle_bin", "description": "Idle-infrastructure bin."},
            {"name": "previous_action_bin", "description": "Previous institution."},
        )
    if state_version == "v2":
        return STATE_V2_FEATURES
    if state_version == "v3":
        return STATE_V3_FEATURES
    raise ValueError(f"Unknown state version {state_version!r}.")


@dataclass(frozen=True)
class QLearningConfig:
    gamma: float = 1.0
    alpha_start: float = 0.25
    alpha_floor: float = 0.03
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.80
    learning_method: str = "one_step_q"
    n_step: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1].")
        if self.alpha_start <= 0 or self.alpha_floor <= 0:
            raise ValueError("Learning rates must be positive.")
        if self.alpha_floor > self.alpha_start:
            raise ValueError("alpha_floor cannot exceed alpha_start.")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Invalid epsilon range.")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon_decay_fraction must lie in (0, 1].")
        if self.learning_method not in {"one_step_q", "n_step_q", "monte_carlo"}:
            raise ValueError(f"Unknown learning method {self.learning_method!r}.")
        if self.n_step <= 0:
            raise ValueError("n_step must be positive.")


class OnlineTabularQPlanner:
    """Epsilon-greedy online Q-learning policy with action masking."""

    name = "online_tabular_q"
    label = "Online tabular Q planner"

    def __init__(
        self,
        *,
        config: QLearningConfig | None = None,
        action_order: Sequence[str] = ACTION_ORDER,
        state_version: str = "v1",
    ):
        self.config = config or QLearningConfig()
        self.action_order = tuple(action_order)
        if state_version not in {"v1", "v2", "v3"}:
            raise ValueError(f"Unknown state version {state_version!r}.")
        self.state_version = state_version
        self.action_to_index = {
            action: index for index, action in enumerate(self.action_order)
        }
        self.q_table: dict[StateKey, np.ndarray] = {}
        self.update_counts: dict[StateKey, np.ndarray] = {}
        self.visit_counts: dict[StateKey, int] = {}

    def state_key(self, observation: CapacityObservation) -> StateKey:
        """Discretize an observation without mutating planner state."""
        if self.state_version == "v3":
            return discretize_observation_v3(observation)
        if self.state_version == "v2":
            return discretize_observation_v2(observation)
        return discretize_observation(observation)

    def _ensure_state(self, state: StateKey) -> None:
        size = len(self.action_order)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(size, dtype=np.float64)
        if state not in self.update_counts:
            self.update_counts[state] = np.zeros(size, dtype=np.int64)
        self.visit_counts.setdefault(state, 0)

    def _state_values(
        self,
        state: StateKey,
        *,
        create: bool,
    ) -> np.ndarray:
        if create:
            self._ensure_state(state)
            return self.q_table[state]
        values = self.q_table.get(state)
        if values is not None:
            return values
        return np.zeros(len(self.action_order), dtype=np.float64)

    def values(self, observation: CapacityObservation) -> np.ndarray:
        state = self.state_key(observation)
        return self._state_values(state, create=True)

    def record_training_visit(self, observation: CapacityObservation) -> None:
        """Record a training-time visit and create the state if needed."""
        state = self.state_key(observation)
        self._ensure_state(state)
        self.visit_counts[state] += 1

    def epsilon_for_episode(self, episode: int, total_episodes: int) -> float:
        if total_episodes <= 1:
            return self.config.epsilon_end
        decay_episodes = max(
            1,
            int(round(total_episodes * self.config.epsilon_decay_fraction)),
        )
        if episode >= decay_episodes:
            return self.config.epsilon_end
        fraction = episode / max(1, decay_episodes - 1)
        return self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * (1.0 - fraction) ** 2

    def choose_action(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
        rng: np.random.Generator,
        *,
        epsilon: float,
        tie_break: str = "stable",
    ) -> str:
        available = [
            action for action in self.action_order if action in available_actions
        ]
        if not available:
            raise RuntimeError("No recognized institutional action is available.")

        state = self.state_key(observation)

        if epsilon > 0 and rng.random() < epsilon:
            return str(rng.choice(available))

        values = self._state_values(state, create=False)
        best_value = max(values[self.action_to_index[action]] for action in available)
        best_actions = [
            action
            for action in available
            if abs(values[self.action_to_index[action]] - best_value) <= 1e-12
        ]
        if tie_break == "random":
            return str(rng.choice(best_actions))
        if tie_break != "stable":
            raise ValueError(f"Unknown tie_break mode {tie_break!r}.")
        # Stable tie-break starts with bilateral bargaining, which acts as the
        # conservative default when available actions are empirically equal.
        return best_actions[0]

    def update(
        self,
        *,
        observation: CapacityObservation,
        action: str,
        reward: float,
        next_observation: CapacityObservation | None,
        next_available_actions: Sequence[str],
        terminated: bool,
    ) -> float:
        if action not in self.action_to_index:
            raise KeyError(f"Unknown planner action {action!r}.")

        state = self.state_key(observation)
        self._ensure_state(state)
        action_index = self.action_to_index[action]

        if terminated or next_observation is None:
            target = float(reward)
        else:
            next_state = self.state_key(next_observation)
            self._ensure_state(next_state)
            next_indices = [
                self.action_to_index[next_action]
                for next_action in next_available_actions
                if next_action in self.action_to_index
            ]
            future = (
                max(self.q_table[next_state][index] for index in next_indices)
                if next_indices
                else 0.0
            )
            target = float(reward) + self.config.gamma * future

        self.update_counts[state][action_index] += 1
        n_updates = int(self.update_counts[state][action_index])
        alpha = max(
            self.config.alpha_floor,
            self.config.alpha_start / np.sqrt(n_updates),
        )
        old_value = self.q_table[state][action_index]
        td_error = target - old_value
        self.q_table[state][action_index] = old_value + alpha * td_error
        return float(td_error)

    def update_toward(
        self,
        *,
        observation: CapacityObservation,
        action: str,
        target: float,
    ) -> float:
        if action not in self.action_to_index:
            raise KeyError(f"Unknown planner action {action!r}.")
        state = self.state_key(observation)
        self._ensure_state(state)
        action_index = self.action_to_index[action]
        self.update_counts[state][action_index] += 1
        n_updates = int(self.update_counts[state][action_index])
        alpha = max(
            self.config.alpha_floor,
            self.config.alpha_start / np.sqrt(n_updates),
        )
        old_value = self.q_table[state][action_index]
        td_error = float(target) - old_value
        self.q_table[state][action_index] = old_value + alpha * td_error
        return float(td_error)

    def n_step_target(
        self,
        transitions: Sequence[Transition],
        *,
        start_index: int,
        n_step: int,
    ) -> float:
        rewards = [
            float(transitions[index]["reward"])
            for index in range(start_index, min(len(transitions), start_index + n_step))
        ]
        target = sum(rewards)
        bootstrap_index = start_index + n_step
        if bootstrap_index < len(transitions):
            next_observation = transitions[bootstrap_index - 1]["next_observation"]
            next_available = transitions[bootstrap_index - 1]["next_available_actions"]
            terminated = bool(transitions[bootstrap_index - 1]["terminated"])
            if not terminated and next_observation is not None:
                next_values = self._state_values(
                    self.state_key(next_observation),
                    create=True,
                )
                next_indices = [
                    self.action_to_index[action]
                    for action in next_available
                    if action in self.action_to_index
                ]
                if next_indices:
                    target += self.config.gamma * max(
                        next_values[index] for index in next_indices
                    )
        return float(target)

    def update_episode_n_step(
        self,
        transitions: Sequence[Transition],
        *,
        n_step: int,
    ) -> list[float]:
        return [
            self.update_toward(
                observation=transition["observation"],
                action=str(transition["action"]),
                target=self.n_step_target(
                    transitions,
                    start_index=index,
                    n_step=n_step,
                ),
            )
            for index, transition in enumerate(transitions)
        ]

    def monte_carlo_returns(
        self,
        transitions: Sequence[Transition],
    ) -> list[float]:
        returns = [0.0] * len(transitions)
        running = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            running = float(transitions[index]["reward"]) + self.config.gamma * running
            returns[index] = running
        return returns

    def update_episode_monte_carlo(
        self,
        transitions: Sequence[Transition],
    ) -> list[float]:
        returns = self.monte_carlo_returns(transitions)
        return [
            self.update_toward(
                observation=transition["observation"],
                action=str(transition["action"]),
                target=target,
            )
            for transition, target in zip(transitions, returns)
        ]

    def diagnostics(
        self,
        observation: CapacityObservation,
        available_actions: Sequence[str],
    ) -> dict[str, object]:
        state = self.state_key(observation)
        values = self._state_values(state, create=False)
        available = [
            action for action in self.action_order if action in available_actions
        ]
        return {
            "q_state_key": ",".join(map(str, state)),
            "q_state_seen": int(self.visit_counts.get(state, 0) > 0),
            "q_state_visits": int(self.visit_counts.get(state, 0)),
            **{
                f"q_{action}": float(values[self.action_to_index[action]])
                for action in self.action_order
            },
            **{
                f"available_{action}": int(action in available)
                for action in self.action_order
            },
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "action_order": list(self.action_order),
            "state_version": self.state_version,
            "config": {
                "gamma": self.config.gamma,
                "alpha_start": self.config.alpha_start,
                "alpha_floor": self.config.alpha_floor,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay_fraction": self.config.epsilon_decay_fraction,
                "learning_method": self.config.learning_method,
                "n_step": self.config.n_step,
            },
            "q_table": {
                ",".join(map(str, state)): values.tolist()
                for state, values in self.q_table.items()
            },
            "update_counts": {
                ",".join(map(str, state)): values.tolist()
                for state, values in self.update_counts.items()
            },
            "visit_counts": {
                ",".join(map(str, state)): int(value)
                for state, value in self.visit_counts.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OnlineTabularQPlanner":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        planner = cls(
            config=QLearningConfig(**payload["config"]),
            action_order=payload["action_order"],
            state_version=payload.get("state_version", "v1"),
        )
        planner.q_table = {
            tuple(int(value) for value in key.split(",")): np.asarray(
                values, dtype=np.float64
            )
            for key, values in payload["q_table"].items()
        }
        planner.update_counts = {
            tuple(int(value) for value in key.split(",")): np.asarray(
                values, dtype=np.int64
            )
            for key, values in payload.get("update_counts", {}).items()
        }
        planner.visit_counts = {
            tuple(int(value) for value in key.split(",")): int(value)
            for key, value in payload.get("visit_counts", {}).items()
        }
        for state in planner.q_table:
            planner._ensure_state(state)
        return planner
