from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    num_nodes: int = 10
    time_slots: int = 80
    slot_seconds: float = 1.0
    min_tasks_per_slot: int = 0
    max_tasks_per_slot: int = 28
    mean_tasks_per_slot: float = 8.0
    task_size_mb: tuple[float, float] = (4.0, 32.0)
    cycles_per_mb: tuple[float, float] = (0.35, 1.10)
    link_ar: float = 0.86
    capacity_ar: float = 0.90
    burst_start_probability: float = 0.08
    burst_end_probability: float = 0.45
    burst_multiplier: float = 2.0
    delay_reward_scale: float = 1.0
    energy_reward_scale: float = 60.0
    reward_clip: float = 8.0
