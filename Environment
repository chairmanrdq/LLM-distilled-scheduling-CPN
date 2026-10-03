from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import EnvironmentConfig


@dataclass(frozen=True)
class Task:
    size_mb: float
    cycles_g: float


class ComputingPowerNetworkEnv:
    """Access-domain simulator for delay-energy scheduling."""

    GLOBAL_FEATURES = 3
    NODE_FEATURES = 7

    def __init__(self, config: EnvironmentConfig, topology_seed: int = 17):
        self.cfg = config
        self.n_nodes = config.num_nodes
        self.action_dim = self.n_nodes
        self.state_dim = self.GLOBAL_FEATURES + self.NODE_FEATURES * self.n_nodes
        self.topology_rng = np.random.default_rng(topology_seed)
        self._build_topology()
        self.rng = np.random.default_rng(topology_seed)
        self.tasks: List[List[Task]] = []
        self.t = 0
        self.n = 0

    def _build_topology(self) -> None:
        rng = self.topology_rng
        self.base_capacity = rng.uniform(12.0, 30.0, self.n_nodes)
        self.base_rate = rng.uniform(80.0, 220.0, self.n_nodes)
        self.propagation_s = rng.uniform(1.0, 8.0, self.n_nodes) / 1000.0
        self.idle_power_w = rng.uniform(28.0, 58.0, self.n_nodes)
        self.dynamic_power_w = rng.uniform(55.0, 125.0, self.n_nodes)
        self.tx_power_w = 0.5

    def sample_episode(self, seed: int, load_scale: float = 1.0) -> List[List[Task]]:
        rng = np.random.default_rng(seed)
        tasks: List[List[Task]] = []
        burst_active = False
        for _ in range(self.cfg.time_slots):
            if burst_active:
                burst_active = rng.random() >= self.cfg.burst_end_probability
            else:
                burst_active = rng.random() < self.cfg.burst_start_probability
            multiplier = self.cfg.burst_multiplier if burst_active else 1.0
            lam = self.cfg.mean_tasks_per_slot * load_scale * multiplier
            count = int(
                np.clip(
                    rng.poisson(lam),
                    self.cfg.min_tasks_per_slot,
                    self.cfg.max_tasks_per_slot,
                )
            )
            size = rng.uniform(*self.cfg.task_size_mb, count)
            density = rng.uniform(*self.cfg.cycles_per_mb, count)
            tasks.append([Task(float(s), float(s * d)) for s, d in zip(size, density)])
        return tasks

    def reset(
        self,
        seed: int,
        tasks: Optional[List[List[Task]]] = None,
        load_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.rng = np.random.default_rng(seed)
        self.tasks = tasks if tasks is not None else self.sample_episode(seed + 10_000, load_scale)
        self.queue_g = np.zeros(self.n_nodes, dtype=np.float64)
        self.link_factor = np.ones(self.n_nodes, dtype=np.float64)
        self.capacity_factor = np.ones(self.n_nodes, dtype=np.float64)
        self.t = 0
        self.n = 0
        self._advance_exogenous_state(initial=True)
        self._skip_empty_slots()
        if self.t >= len(self.tasks):
            raise ValueError("The sampled episode contains no tasks")
        return self._state(), self.valid_action_mask()

    def _advance_exogenous_state(self, initial: bool = False) -> None:
        innovation_scale = 0.07 if not initial else 0.025
        self.link_factor = np.clip(
            self.cfg.link_ar * self.link_factor
            + (1.0 - self.cfg.link_ar)
            + self.rng.normal(0.0, innovation_scale, self.n_nodes),
            0.55,
            1.30,
        )
        self.capacity_factor = np.clip(
            self.cfg.capacity_ar * self.capacity_factor
            + (1.0 - self.cfg.capacity_ar)
            + self.rng.normal(0.0, innovation_scale, self.n_nodes),
            0.65,
            1.20,
        )

    @property
    def capacity(self) -> np.ndarray:
        return self.base_capacity * self.capacity_factor

    def effective_rates(self) -> np.ndarray:
        return self.base_rate * self.link_factor

    def valid_action_mask(self) -> np.ndarray:
        return (self.capacity > 1e-6) & (self.effective_rates() > 1e-6)

    def _estimates(self, task: Task) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rates = self.effective_rates()
        capacity = self.capacity
        transmission_time = task.size_mb / np.maximum(rates, 1e-6)
        transmission = transmission_time + self.propagation_s
        waiting = self.queue_g / np.maximum(capacity, 1e-6)
        computing = task.cycles_g / np.maximum(capacity, 1e-6)
        delay = transmission + waiting + computing
        compute_energy_per_gcycle = self.dynamic_power_w / np.maximum(self.base_capacity, 1e-6)
        energy = self.tx_power_w * transmission_time + compute_energy_per_gcycle * task.cycles_g
        return delay, energy, rates

    def _state(self) -> np.ndarray:
        task = self.tasks[self.t][self.n]
        delay, energy, rates = self._estimates(task)
        capacity = self.capacity
        queue_load = np.log1p(self.queue_g / np.maximum(capacity, 1e-6)) / np.log(9.0)
        delay_score = np.log1p(delay / self.cfg.delay_reward_scale)
        energy_score = np.log1p(energy / self.cfg.energy_reward_scale)
        node_features = np.stack(
            [
                queue_load,
                capacity / 30.0,
                rates / 220.0,
                self.propagation_s / 0.008,
                self.dynamic_power_w / 125.0,
                delay_score,
                energy_score,
            ],
            axis=1,
        ).reshape(-1)
        global_features = np.array(
            [
                task.size_mb / self.cfg.task_size_mb[1],
                task.cycles_g / (self.cfg.task_size_mb[1] * self.cfg.cycles_per_mb[1]),
                len(self.tasks[self.t]) / self.cfg.max_tasks_per_slot,
            ],
            dtype=np.float64,
        )
        return np.concatenate([global_features, node_features]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, float], np.ndarray]:
        mask = self.valid_action_mask()
        if action < 0 or action >= self.n_nodes or not mask[action]:
            raise ValueError(f"Invalid action {action}; valid mask={mask.astype(int).tolist()}")
        task = self.tasks[self.t][self.n]
        delays, energies, _ = self._estimates(task)
        delay = float(delays[action])
        energy = float(energies[action])
        queue_before = float(self.queue_g[action])
        self.queue_g[action] += task.cycles_g

        reward = np.array(
            [
                -np.clip(delay / self.cfg.delay_reward_scale, 0.0, self.cfg.reward_clip),
                -np.clip(energy / self.cfg.energy_reward_scale, 0.0, self.cfg.reward_clip),
            ],
            dtype=np.float32,
        )
        info = {
            "delay": delay,
            "energy": energy,
            "node": float(action),
            "queue_before": queue_before,
            "total_queue": float(self.queue_g.sum()),
        }

        self.n += 1
        if self.n >= len(self.tasks[self.t]):
            self._finish_slot()
            self.t += 1
            self.n = 0
            self._skip_empty_slots()
        done = self.t >= len(self.tasks)
        if done:
            return np.zeros(self.state_dim, dtype=np.float32), reward, True, info, np.ones(self.n_nodes, dtype=bool)
        return self._state(), reward, False, info, self.valid_action_mask()

    def _finish_slot(self) -> None:
        self.queue_g = np.maximum(self.queue_g - self.capacity * self.cfg.slot_seconds, 0.0)
        self._advance_exogenous_state()

    def _skip_empty_slots(self) -> None:
        while self.t < len(self.tasks) and len(self.tasks[self.t]) == 0:
            self._finish_slot()
            self.t += 1


def state_node_matrix(state: np.ndarray, num_nodes: int) -> np.ndarray:
    start = ComputingPowerNetworkEnv.GLOBAL_FEATURES
    return np.asarray(state[start:], dtype=np.float64).reshape(
        num_nodes, ComputingPowerNetworkEnv.NODE_FEATURES
    )
