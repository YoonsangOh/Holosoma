# BeyondMimic Training Run — Stairs Scene

> **Status:** Training in progress (started 2026-02-23).
> The "Training Results" section at the bottom will be updated after the 2000-iteration run completes.

---

## 1. What Is BeyondMimic?

BeyondMimic (referenced throughout the holosoma codebase as **Whole Body Tracking / WBT**) is a motion-imitation algorithm for humanoid robots that goes beyond simple flat-ground kinematic replay. The key idea is to train a **physics-based control policy** that reproduces full-body human motions — including limb trajectories, orientations, and velocities — inside a **real 3D scene** (e.g., stairs, slopes, crawl-spaces), rather than on an idealized flat plane.

In holosoma, BeyondMimic is implemented as the `WholeBodyTrackingManager` environment class, using **Proximal Policy Optimization (PPO)** as the RL algorithm and IsaacSim as the physics simulator.

---

## 2. Training Setup for This Run

### Data

| Item | Path |
|---|---|
| Motion reference | `data/stairs/motion_for_holosoma.npz` |
| Scene mesh | `data/stairs/scene_mesh_gravity_aligned.obj` |

**Motion file details:**
- 465 frames × 50 fps → ~9.3 seconds of stair-climbing motion
- 29-DOF joint positions + root pose (36 values per frame)
- Full-body poses for 51 MuJoCo links
- Originally retargeted from real human video via Real2Sim pipeline

**Scene file:**
- OBJ mesh of the stairs geometry, gravity-aligned (z-up)
- Loaded as a static trimesh terrain via the `terrain:terrain-load-obj` preset

### Environment Configuration

| Parameter | Value |
|---|---|
| Simulator | IsaacSim (NVIDIA Omniverse) |
| Number of parallel envs | 2048 |
| Max episode length | 10.0 s (500 steps at 50 Hz control) |
| Environment spacing | 0.0 m (all envs share the same stairs scene) |
| Control decimation | 20 (50 Hz control, 1000 Hz physics) |
| Robot | Unitree G1 (29-DOF) |
| Self-collisions | Enabled |
| Robot init height | 0.76 m |

### Algorithm: PPO

PPO (Proximal Policy Optimization) collects rollouts from all parallel environments, then performs multiple gradient update passes before collecting new data.

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO |
| Actor learning rate | 1e-5 (AdamW, weight_decay=0.0) |
| Critic learning rate | 1e-4 (AdamW, weight_decay=0.0) |
| Clip parameter (ε) | 0.2 |
| Discount factor (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Entropy coefficient | 0.005 |
| Value loss coefficient | 1.0 |
| Max gradient norm | 1.0 |
| Learning epochs per iter | 8 |
| Mini-batches per epoch | 4 |
| Steps per env per iter | 24 |
| LR schedule | Adaptive (KL target = 0.01) |
| Total iterations | 2000 |
| Checkpoint save interval | 200 |
| Symmetry loss | Disabled (`use_symmetry=False`) |
| Initial noise std | 1.0 |

### Policy Network Architecture

Both the actor and critic use a 3-layer MLP with ELU activations:

```
Input → Linear(512) → ELU → Linear(256) → ELU → Linear(128) → ELU → Output
```

- **Actor input:** 154-dimensional observation → 29-dimensional joint position output
- **Critic input:** 286-dimensional observation → scalar value estimate

The actor outputs delta joint positions (action_scale = 1.0), applied on top of the default pose.

---

## 3. Observation Space (154 for Actor, 286 for Critic)

### Actor Observations (154-dim, with noise during training)

| Term | Dim | Description |
|---|---|---|
| `actions` | 29 | Previous joint position actions |
| `base_ang_vel` | 3 | Base angular velocity (noise σ=0.2) |
| `dof_pos` | 29 | Current joint positions (noise σ=0.01) |
| `dof_vel` | 29 | Current joint velocities (noise σ=0.5) |
| `motion_command` | 58 | Reference body poses from motion clip (14 bodies × position+orientation) |
| `motion_ref_ori_b` | 6 | Reference orientation of torso in robot base frame (noise σ=0.05) |

### Critic Observations (286-dim, no noise)

Includes all actor observations plus:

| Term | Dim | Description |
|---|---|---|
| `base_lin_vel` | 3 | Base linear velocity (ground truth, no noise) |
| `motion_ref_pos_b` | 3 | Reference position of torso in base frame |
| `robot_body_pos_b` | 42 | Positions of 14 tracked bodies in base frame |
| `robot_body_ori_b` | 84 | Orientations of 14 tracked bodies (6D rotation) in base frame |

The critic has access to privileged information (true velocities, full body state) that the actor does not, implementing asymmetric actor-critic for WBT.

---

## 4. Reward Structure

All rewards are averaged over the episode and logged to WandB.

### Tracking Rewards (positive)

| Reward Term | Formula | Weight |
|---|---|---|
| Global reference position error | `exp(-‖pos_ref - pos_robot‖² / σ²)`, σ=0.3 | 0.5 |
| Global reference orientation error | `exp(-‖ori_ref - ori_robot‖² / σ²)`, σ=0.4 | 0.5 |
| Relative body position error | Position error in torso frame, σ=0.3 | 1.0 |
| Relative body orientation error | Orientation error in torso frame, σ=0.4 | 1.0 |
| Body linear velocity tracking | exp(-‖vel_ref - vel_robot‖² / σ²), σ=1.0 | 1.0 |
| Body angular velocity tracking | exp(-‖ang_vel_ref - ang_vel_robot‖² / σ²), σ=π | 1.0 |

All tracking rewards are computed over the 14 tracked body links:
`pelvis, left_hip_roll_link, left_knee_link, left_ankle_roll_link, right_hip_roll_link, right_knee_link, right_ankle_roll_link, torso_link, left_shoulder_roll_link, left_elbow_link, left_wrist_yaw_link, right_shoulder_roll_link, right_elbow_link, right_wrist_yaw_link`

### Regularization Penalties (negative)

| Penalty Term | Weight |
|---|---|
| Action rate L2 (`‖a_t - a_{t-1}‖²`) | −0.1 |
| DOF position limit violation | −100.0 (soft limit at 90% of range) |
| Undesired contacts (non-foot/wrist/ankle body parts) | −0.5 |

---

## 5. Motion Command System

At each step, the `MotionCommand` manager:
1. Samples a frame from the reference motion clip (`motion_for_holosoma.npz`)
2. Provides the reference body poses as the `motion_command` observation
3. Computes tracking errors for the reward function
4. Resets to a random frame in the clip at episode start (with small noise added to the initial pose)

**Initial pose noise:**
- Root position: ±5 cm (x,y), ±1 cm (z)
- Root rotation: ±0.1 rad (roll/pitch), ±0.2 rad (yaw)
- Root linear velocity: ±10 cm/s (x,y), ±5 cm/s (z)
- Root angular velocity: ±0.1 rad/s (all axes)
- Joint positions: ±0.1 rad

---

## 6. Scene Loading

The stairs OBJ mesh (`scene_mesh_gravity_aligned.obj`) is loaded via the `terrain:terrain-load-obj` preset:

1. The OBJ file is parsed as a trimesh
2. In IsaacSim: converted to a USD-compatible physics mesh with static friction=1.0, dynamic friction=1.0, restitution=0.0
3. All 2048 environments are placed at the **same location** (env_spacing=0.0), so every robot trains on the identical stairs geometry

This differs from procedural terrain training where each environment might see a different terrain sub-tile.

---

## 7. Training Command Used

```bash
WANDB_API_KEY="54d5951df3502e196e3a1b895e227e9969fc8e7b" \
OMNI_KIT_ACCEPT_EULA=1 \
CUDA_VISIBLE_DEVICES=0 \
PATH="$HOME/.holosoma_deps/miniconda3/envs/hssim/bin:$PATH" \
~/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    terrain:terrain-load-obj \
    logger:wandb \
    --terrain.terrain-term.obj-file-path="/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/scene_mesh_gravity_aligned.obj" \
    --command.setup-terms.motion-command.params.motion-config.motion-file="/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/motion_for_holosoma.npz" \
    --simulator.config.scene.env-spacing=0.0 \
    --training.num-envs 2048 \
    --training.project beyondmimic \
    --algo.config.num-learning-iterations 2000 \
    --algo.config.save-interval 200 \
    --logger.video.enabled True \
    --logger.video.interval 5 \
    --logger.headless-recording True
```

**Note on `CUDA_VISIBLE_DEVICES=0`:** IsaacSim uses Vulkan/Omniverse for GPU enumeration which conflicts with `CUDA_VISIBLE_DEVICES`. The launcher emits warnings ("CUDA being in bad state") but training proceeds successfully — IsaacSim selects a usable GPU via its own enumeration independent of this variable.

---

## 8. WandB Logging

- **Project:** `beyondmimic`
- **Run URL:** https://wandb.ai/5yoondori-seoul-national-university/beyondmimic/runs/qo69q8mz
- **Videos:** Logged every ~5 completed episodes for env 0 ≈ once every 100 training iterations (~20 videos total over 2000 iterations). Rendered at 640×360, h264.
- **Metrics logged:** All reward terms, evaluation metrics (action smoothness, foot slippage, contact force, motion tracking score), policy noise std, learning rates

---

## 9. Checkpoint Locations

Checkpoints are saved every 200 iterations to:

```
logs/beyondmimic/20260223_g1_29dof_wbt_manager/
├── model_00000.pt    ← initial weights (before any training)
├── model_00200.pt
├── model_00400.pt
├── ...
└── model_02000.pt    ← final checkpoint
```

Each `.pt` file contains:
- `actor_model_state_dict`: actor MLP weights
- `critic_model_state_dict`: critic MLP weights
- `actor_optimizer_state_dict` / `critic_optimizer_state_dict`
- Training metadata (iteration, mean reward, etc.)

---

## 10. Training Results

> ⏳ **Training in progress.** This section will be updated after the 2000-iteration run completes (~2.5 hours from start).

**To monitor live:**
https://wandb.ai/5yoondori-seoul-national-university/beyondmimic/runs/qo69q8mz

**Log file:** `/tmp/holosoma_beyondmimic.log`

---

*(This document was written during the training run on 2026-02-23.)*
