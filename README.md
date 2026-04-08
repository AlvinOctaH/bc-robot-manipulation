# Behavior Cloning for Robot Manipulation

Imitation learning pipeline implementing Behavior Cloning (BC) for a robotic lift task using Robosuite and PyTorch. Built as a portfolio project demonstrating core concepts in robot learning relevant to surgical autonomy research.

## Overview

This project implements the full BC pipeline:
1. **Expert demonstration collection** — scripted policy collects (observation, action) pairs
2. **BC training** — MLP policy trained via supervised learning on expert data
3. **Policy evaluation** — learned policy evaluated on unseen episodes

## Results

| Experiment | Demos | Obs Norm | Noise Injection | Success Rate |
|---|---|---|---|---|
| Baseline | 50 | ✗ | ✗ | 0% |
| + Normalization | 200 | ✓ | ✗ | 15% |
| + Noise Injection | 500 | ✓ | ✓ | **80%** |

## Key Findings

**Distribution shift** is the primary failure mode in BC. When the policy encounters a state slightly outside the training distribution, errors compound — a small positional error leads to incorrect gripper timing, causing grasp failure.

**Observation normalization** is critical when input features have different scales (e.g. `eef_pos` ~0.8 vs `gripper_qpos` ~0.001). Without normalization, large-scale features dominate gradients and the policy becomes blind to small but important features like gripper state.

**Noise injection** during demonstration collection improves robustness by exposing the policy to slightly perturbed states, reducing distribution shift at evaluation time.

## Setup

```bash
conda create -n imitation_learning python=3.10 -y
conda activate imitation_learning
pip install robosuite
pip install robomimic --no-deps
pip install h5py psutil tqdm termcolor tensorboard tensorboardX imageio imageio-ffmpeg matplotlib torch torchvision
```

## Usage

```bash
# 1. Collect expert demonstrations
python scripts/collect_demos.py

# 2. Train BC policy
python scripts/train_bc.py

# 3. Evaluate
python scripts/evaluate_bc.py
```

## Architecture

**Policy:** 3-layer MLP (256 hidden units, ReLU activations, Tanh output)

**Observation space (15-dim):**
- `eef_pos` (3) — end-effector position
- `eef_quat` (4) — end-effector orientation
- `gripper_qpos` (2) — gripper state
- `cube_pos` (3) — object position
- `gripper_to_cube_pos` (3) — relative distance

**Action space (7-dim):** OSC_POSE delta commands + gripper

## Limitations & Future Work

- BC suffers from compounding errors (distribution shift) — DAgger would address this by iteratively collecting data from failure states
- State-based observations only — visual BC (image input) would be more relevant to real surgical settings, as explored in SuFIA-BC
- Task complexity is limited — surgical peg transfer requires multi-stage coordination across two arms (dVRK PSM)

## Connection to Surgical Robotics

This pipeline mirrors the core methodology of [Orbit-Surgical](https://orbit-surgical.github.io/) and [SuFIA-BC](https://orbit-surgical.github.io/sufia-bc/) — both use scripted expert demonstrations + behavior cloning for surgical subtask learning. The key difference is the simulation environment (Robosuite vs Isaac Lab) and robot platform (Panda vs dVRK PSM).