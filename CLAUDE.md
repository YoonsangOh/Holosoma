# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Holosoma** is a humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots (Unitree G1, Booster T1). It consists of three packages:

- `src/holosoma/` — Core RL training (PPO, FastSAC) across multiple simulators
- `src/holosoma_inference/` — Sim-to-sim and sim-to-real policy deployment
- `src/holosoma_retargeting/` — Human mocap to robot motion conversion

## Environment Setup

Each component has its own isolated environment, activated via source scripts:

```bash
# IsaacGym training
bash scripts/setup_isaacgym.sh
source scripts/source_isaacgym_setup.sh

# IsaacSim training (Ubuntu 22.04+ required)
bash scripts/setup_isaacsim.sh
source scripts/source_isaacsim_setup.sh

# MuJoCo/MJWarp training
bash scripts/setup_mujoco.sh
source scripts/source_mujoco_setup.sh

# Inference/deployment
bash scripts/setup_inference.sh
source scripts/source_inference_setup.sh

# Motion retargeting
bash scripts/setup_retargeting.sh
source scripts/source_retargeting_setup.sh
```

## Common Commands

### Training

```bash
# Activate environment first, then:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb

# Key overrides:
# --training.num-envs 4096
# --training.seed 1
# --algo.config.use-symmetry False
# --reward.terms.tracking-lin-vel.weight 2.5
```

### Evaluation and Replay

```bash
python src/holosoma/holosoma/eval_agent.py --checkpoint=<path_or_wandb_uri>
python src/holosoma/holosoma/replay.py exp:g1-29dof-wbt
```

### Inference (deployment)

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --task.model-path <path_or_wandb_uri> \
    --task.use-joystick \
    --task.interface eth0
```

### Retargeting

```bash
# Single sequence
python examples/robot_retarget.py \
    --data_path demo_data/OMOMO_new \
    --task-type robot_only \
    --task-name sub3_largebox_003 \
    --data_format smplh

# Batch
python examples/parallel_robot_retarget.py \
    --data-dir demo_data/OMOMO_new \
    --task-type robot_only \
    --data_format smplh \
    --save_dir demo_results_parallel/g1/robot_only/omomo
```

### Linting and Type Checking

```bash
# Linting (auto-fix)
ruff check --fix .
ruff format .

# Type checking
mypy src/

# Install and run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Testing

```bash
# All tests
pytest

# Skip IsaacSim tests
pytest -m "not isaacsim"

# Skip multi-GPU tests
pytest -m "not multi_gpu"

# Single test file
pytest src/holosoma/holosoma/agents/modules/tests/test_module_dimensions.py -v

# CI test scripts
bash tests/ci/isaacgym_ci_tests.sh
bash tests/ci/isaacsim_ci_tests.sh
```

## Architecture

### Configuration System

Uses **Tyro** for CLI composition with Pydantic-typed config classes. Experiments are composed from named presets:

```
exp:<name>          # Experiment presets (e.g., g1-29dof-fast-sac, g1-29dof-wbt)
simulator:<name>    # Simulator backend (isaacgym, isaacsim, mjwarp)
terrain:<name>      # Terrain type (terrain-locomotion-mixed, terrain-locomotion-plane)
logger:<name>       # Logger (wandb)
```

Config types live in `config_types/` (Pydantic classes); preset values live in `config_values/`. Overrides use dotted paths matching the config hierarchy.

### Simulator Abstraction

`src/holosoma/holosoma/simulator/` provides a unified interface across IsaacGym, IsaacSim, and MJWarp. IsaacSim requires importing `SimApp` before PyTorch — this is why `train_agent.py` and `eval_utils.py` have late imports (Ruff rule PLC0415 disabled for those files).

### RL Algorithms

`agents/ppo/` and `agents/fast_sac/` extend `agents/base_algo/`. Both share the same `agents/modules/` for neural network building blocks, logging, and data augmentation.

### Inference Pipeline

Policies are exported to ONNX for deployment independence. `holosoma_inference` loads ONNX models and communicates with hardware via SDK bridges (`sdk/` — Unitree SDK2, Booster SDK). Supports both real robot and MuJoCo sim-to-sim evaluation via the same `run_policy.py` entry point.

### Motion Retargeting

`holosoma_retargeting` converts human mocap (OMOMO/smplh, LAFAN/BVH, AMASS/SMPL-X) into robot-executable motions, then `data_conversion/convert_data_format_mj.py` converts outputs to the training data format.

## Code Style

- **Linter/formatter:** Ruff (line length 120, Python 3.8+)
- **Type checker:** MyPy (strict for `torch_utils` module)
- **C/C++/CUDA:** Clang-format (v20.1.3)
- Pre-commit hooks enforce all of the above on commit

## CI/CD

GitHub Actions workflows (`.github/workflows/`):
- `static-checks.yaml` — Ruff + MyPy on PRs (CPU runner)
- `pytest.yaml` — GPU tests on PRs, matrix over isaacgym/isaacsim
- `nightly-training.yaml` — Full training runs nightly with Slack notifications and S3 checkpoint storage

Tests run inside a private ECR Docker image with `--gpus all` and 12GB shared memory.
