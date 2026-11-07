# LC-MOSAC simulation input generator

This package generates topology, task arrivals and attributes, and resource
trajectories. It does not perform model training or scheduling evaluation,
and does not provide trained models or performance results.

The reference CSV dataset is deposited at
https://doi.org/10.5281/zenodo.23118195.
This repository provides generation code, not the complete scheduling software.
Dataset files and their column definitions belong to the data deposit's README;
the instructions below describe how to generate the inputs locally.

## Run

Use Python 3.9 to 3.12 for the pinned NumPy version. From this directory:

```sh
python -m pip install -r requirements.txt
python generate_dataset.py
```

Output is written to `simulation_inputs_v1/`. To use another location:

```sh
python generate_dataset.py --output another_output
```

Existing output files are never overwritten. No GPU, PyTorch, or LLM API is needed.
The generated output directory is excluded by `.gitignore` because the data
are distributed separately. Upload the source files in this directory to GitHub.

## Files

- `environment.py`: simulator implementing topology, task, and resource generation.
- `config.py`: configuration parameters for the simulation environment.
- `generate_dataset.py`: entry point for exporting simulation inputs to CSV files.
- `requirements.txt`: pinned NumPy dependency for the generation configuration.

## Generation and scope

Topology uses `ComputingPowerNetworkEnv._build_topology()` with seed 17.
Tasks use `ComputingPowerNetworkEnv.sample_episode(task_seed, load_scale)`.
Resources use `ComputingPowerNetworkEnv._advance_exogenous_state()` once per
slot, including empty slots. Resource generation is independent of policy actions.
Slot and node indices start at zero.

The export contains 40 episodes: 25 training examples, 5 validation episodes,
and 10 test episodes. Training examples use run seeds 2026 to 2030 and episodes
1, 125, 250, 375, and 500 of a configured 500-episode sequence. For run
seed s and episode e, the resource seed is s + 37e, task seed is s + 37e + 10000,
and load scale is element e-1 of 500 uniform draws on [0.75, 1.25] using seed
s + 777. Validation resource seeds are 310001 to 310005, with load scales
evenly spaced on [0.85, 1.15]. Test resource seeds are 410001 to 410010, with
load scales evenly spaced on [0.80, 1.20]. Evaluation task seeds equal resource
seeds plus 50000.

The dataset contains a fixed collection of simulation input examples.
The training, validation, and test labels identify the subsets defined by the
export configuration. These labels do not indicate that model training or
evaluation is performed by this package. The files contain no scheduling
decisions, trained models, or performance results.

## Output

| File | Contents |
| --- | --- |
| topology.csv | Nominal node capacity, transmission rate, propagation delay and power |
| episodes.csv | Episode identifiers, splits, seeds, load scales and task counts |
| tasks.csv | Task size, computational density and required cycles per arrival |
| resource_dynamics.csv | Resource factors and effective rates/capacities per slot and node |
| metadata.json | Configuration, export scope, runtime NumPy version and source hashes |

The metadata records this export's configuration, scope, NumPy version, and
source hashes. It is not intended to reproduce the metadata of an earlier export.
Source hashes refer to the three Python files in the generator repository,
not to files included in the data deposit.

## Verification

All headers and field values in all four generated CSVs were compared with the
local reference dataset and matched exactly: 10 topology rows, 40 episode rows,
30,051 task rows, and 32,000 resource rows. This export check ran with NumPy
2.3.5; requirements.txt specifies NumPy 1.26.4, the version recorded for the
reference dataset. The generated metadata and this README are not part of the
CSV equivalence claim.
