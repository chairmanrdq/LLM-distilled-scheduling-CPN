"""Export the archived 40-episode simulation-input recipe without training."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import EnvironmentConfig
from environment import ComputingPowerNetworkEnv


def episode_specs():
    rows = []
    for seed in range(2026, 2031):
        scales = np.random.default_rng(seed + 777).uniform(0.75, 1.25, 500)
        for episode in (1, 125, 250, 375, 500):
            resource_seed = seed + episode * 37
            rows.append(dict(split="training_example",
                             episode_id=f"train_seed_{seed}_episode_{episode:03d}",
                             run_seed=seed, training_episode=episode,
                             episode_seed=resource_seed, task_seed=resource_seed + 10000,
                             load_scale=float(scales[episode - 1])))
    for split, count, base, low, high in (
        ("validation", 5, 310001, 0.85, 1.15),
        ("test", 10, 410001, 0.80, 1.20),
    ):
        for i, scale in enumerate(np.linspace(low, high, count)):
            rows.append(dict(split=split, episode_id=f"{split}_{i + 1:02d}",
                             run_seed="", training_episode="", episode_seed=base + i,
                             task_seed=base + i + 50000, load_scale=float(scale)))
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(output):
    names = ("topology.csv", "episodes.csv", "tasks.csv", "resource_dynamics.csv", "metadata.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Output files already exist; choose a new --output directory.")
    cfg = EnvironmentConfig()
    env = ComputingPowerNetworkEnv(cfg, topology_seed=17)
    topology = [dict(node_id=m, base_capacity_gcycles_per_s=float(env.base_capacity[m]),
                     base_rate_mb_per_s=float(env.base_rate[m]),
                     propagation_s=float(env.propagation_s[m]), idle_power_w=float(env.idle_power_w[m]),
                     dynamic_power_w=float(env.dynamic_power_w[m]), transmission_power_w=env.tx_power_w)
                for m in range(cfg.num_nodes)]
    episodes, tasks, resources = [], [], []
    for spec in episode_specs():
        sampled = env.sample_episode(spec["task_seed"], spec["load_scale"])
        episodes.append(dict(**spec, time_slots=cfg.time_slots, total_tasks=sum(map(len, sampled))))
        env.rng = np.random.default_rng(spec["episode_seed"])
        env.link_factor = np.ones(cfg.num_nodes)
        env.capacity_factor = np.ones(cfg.num_nodes)
        for slot, arrivals in enumerate(sampled):
            # Advance every slot, including empty ones, without scheduling actions.
            env._advance_exogenous_state(initial=(slot == 0))
            for m in range(cfg.num_nodes):
                resources.append(dict(split=spec["split"], episode_id=spec["episode_id"],
                                      slot=slot, node_id=m, link_factor=float(env.link_factor[m]),
                                      capacity_factor=float(env.capacity_factor[m]),
                                      effective_rate_mb_per_s=float(env.effective_rates()[m]),
                                      effective_capacity_gcycles_per_s=float(env.capacity[m])))
            for i, task in enumerate(arrivals):
                tasks.append(dict(split=spec["split"], episode_id=spec["episode_id"], slot=slot,
                                  task_index=i, task_size_mb=task.size_mb,
                                  compute_density_gcycles_per_mb=task.cycles_g / task.size_mb,
                                  required_cycles_g=task.cycles_g))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(names, (topology, episodes, tasks, resources)):
        write_csv(output / name, rows)
    root = Path(__file__).resolve().parent
    metadata = dict(dataset_version="1.0.0", export_tool_version="1.0.0",
                    description="Policy-independent simulation inputs; no labels, policies or results.",
                    topology_seed=17, environment_config=asdict(cfg),
                    split_counts={s: sum(r["split"] == s for r in episodes)
                                  for s in ("training_example", "validation", "test")},
                    numpy_version=np.__version__,
                    source_sha256={n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                                   for n in ("config.py", "environment.py", "generate_dataset.py")},
                    generation={"topology_function": "ComputingPowerNetworkEnv._build_topology",
                                "task_function": "ComputingPowerNetworkEnv.sample_episode",
                                "resource_function": "ComputingPowerNetworkEnv._advance_exogenous_state"},
                    training_recipe={"run_seeds": list(range(2026, 2031)), "sequence_length": 500,
                                     "selected_episodes": [1, 125, 250, 375, 500],
                                     "resource_seed": "s + 37 * e", "task_seed": "s + 37 * e + 10000",
                                     "load_scale": "default_rng(s + 777).uniform(0.75, 1.25, 500)[e - 1]"},
                    validation_recipe={"resource_seeds": [310001, 310005], "load_scale_range": [0.85, 1.15]},
                    test_recipe={"resource_seeds": [410001, 410010], "load_scale_range": [0.80, 1.20]},
                    notes=["Training, validation and test are input subset names; no models are fitted.",
                           "Evaluation task seeds equal resource seeds plus 50000; load scales are evenly spaced.",
                           "Resource trajectories advance once per slot, including empty slots.",
                           "Indices for slots, nodes and tasks within a slot start at zero.",
                           "CSV data are unchanged from the reference export; documentation and source hashes are updated."])
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(episodes)} episodes, {len(tasks)} tasks and {len(resources)} resource rows to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "simulation_inputs_v1")
    generate(parser.parse_args().output)
