1# Holosoma Project Progress

---

## 2026-03-12

### Goal
Run motion-tracking training for `data/unitree_lafan_29dof` on the current 3090 server, verify end-to-end learning with W&B logging, and establish a safe multi-clip training path without rewriting the core training loop.

---

### 1. Runtime Status and Backend Decision

- Checked GPU occupancy with `nvidia-smi`; GPUs 3/4/5 were effectively idle.
- Attempted IsaacSim on GPU 3 first because WBT is primarily configured for IsaacSim.
- IsaacSim startup and W&B login succeeded, but this server reported repeated Omniverse warnings:
  - `Skipping NVIDIA GPU due CUDA being in bad state`
  - missing `libGLU.so.1` in the RTX neuray plugin path
- Practical result: IsaacSim was launching but falling back to a CPU-heavy path and not finishing short verification runs in reasonable time.

**Conclusion:** kept IsaacSim compatibility work, but used MuJoCo classic for the successful end-to-end verification run on this server.

---

### 2. Compatibility / Launch Fixes Added

**Added:** `scripts/run_train_agent_with_isaaclab_site.py`
- Allows Holosoma to run with the `holosoma_yoonsang` Python environment while late-loading IsaacLab/IsaacSim site-packages from `env_isaaclab`.
- Normalizes `scene.scene_files=[]` and `scene.rigid_objects=[]` back to `None` before training so Tyro parsing works without breaking IsaacSim runtime semantics.

**Adjusted:** `src/holosoma/holosoma/config_types/video.py`
- Relaxed the `VideoConfig.camera` annotation to avoid a pydantic v2 discriminated-union parsing failure on this server.

**Adjusted:** `src/holosoma/holosoma/config_types/simulator.py`
- Restored `scene_files` / `rigid_objects` to optional list types while preserving a Tyro-friendly default.

**Adjusted:** `src/holosoma/holosoma/simulator/isaacsim/isaacsim.py`
- Added `HOLOSOMA_USD_CONVERSION_ROOT` support so URDF->USD conversion cache can be redirected to local storage instead of the NAS-backed robot asset directory.

---

### 3. Successful Single-Clip Training Run

**Backend used:** `simulator:mujoco`

**Why MuJoCo:** IsaacSim could not complete short runs reliably on this machine due the GPU/runtime issues above, while MuJoCo could execute the same WBT future-motion PPO stack without changing the core training code.

**Command shape:**
- preset: `exp:g1-29dof-wbt-future-motion`
- simulator: `mujoco`
- logger: `wandb`
- envs: `1`
- iterations: `2`
- checkpoint interval: `1`
- unsupported IsaacSim-only randomizers skipped via `--randomization.ignore-unsupported True`

**Motion file trained:**
- `data/unitree_lafan_29dof/lafan_walk_and_dance/env_10_walk3_subject3.npz`

**Artifacts produced:**
- run dir: `logs/LAFANMotionTracking/20260312_lafan_walk3_subject3_mujoco_tiny2_20260312`
- checkpoints:
  - `model_00000.pt`
  - `model_00001.pt`
- exported ONNX:
  - `model_00000.onnx`
  - `model_00001.onnx`
- config:
  - `holosoma_config.yaml`
- log:
  - `train.log`

**W&B run:**
- project: `LAFANMotionTracking`
- run name: `lafan_walk3_subject3_mujoco_tiny2_20260312`
- URL: `https://wandb.ai/5yoondori-seoul-national-university/LAFANMotionTracking/runs/jl74hi3q`

**Observed metric snapshot from the completed run:**
- Iteration 0:
  - `motion_tracking_global = 2.7327`
  - `motion_tracking_local = 2.7341`
- Iteration 1:
  - `motion_tracking_global = 2.7253`
  - `motion_tracking_local = 2.7237`

This confirms the full train loop, checkpoint saving, ONNX export, and W&B logging all completed successfully.

---

### 4. Multi-Clip Sequential Policy Training

Because the existing `MotionLoader` still accepts a single `motion_file`, I did **not** modify the core WBT loader. Instead, I added a thin orchestration script that chains the latest checkpoint into the next run:

**Added:** `scripts/train/train_lafan_multiclip_sequential.sh`

**Method:**
1. Train one clip.
2. Take the newest `model_*.pt`.
3. Launch the next clip with `--training.checkpoint <previous_ckpt>`.
4. Repeat.

This gives one continually updated policy across clips while keeping Holosoma’s native `train_agent.py` untouched.

**Validation run completed (3 clips):**
- clip 0: `env_10_walk3_subject3.npz`
- clip 1: `env_11_dance1_subject2.npz`
- clip 2: `env_12_dance2_subject5.npz`

**Chain record:**
- `logs/LAFANMotionTracking/multiclip_20260312_155226/checkpoint_chain.txt`

**Final chained checkpoint:**
- `logs/LAFANMotionTracking/20260312_lafan_multiclip_seq_mujoco_02_env_12_dance2_subject5/model_00001.pt`

**W&B runs created:**
- `lafan_multiclip_seq_mujoco_00_env_10_walk3_subject3`
- `lafan_multiclip_seq_mujoco_01_env_11_dance1_subject2`
- `lafan_multiclip_seq_mujoco_02_env_12_dance2_subject5`

This verifies that sequential multi-clip continual training works with the current codebase.

---

### 5. Recommended Path Forward

- Short term:
  - Continue using `scripts/train/train_lafan_multiclip_sequential.sh` for a full curriculum over `lafan_walk_and_dance` or `lafan_replay_data`.
  - Keep the run on MuJoCo/MuJoCo+resume for reliable iteration on this server.
- Medium term:
  - Restore a clean IsaacSim runtime on a server where Omniverse sees the target 3090 as healthy.
  - Re-run the same clip chain with IsaacSim once `CUDA bad state` / `libGLU.so.1` issues are resolved.
- Long term:
  - Add native multi-file motion sampling to `MotionLoader` / `MotionCommand` so a single run can sample clips directly instead of shell-level curriculum.

---

### 6. Native Multi-Clip WBT in Holosoma + Full Training Launch

**Goal:** replace shell-level sequential clip chaining with native multi-clip sampling inside Holosoma’s WBT command path, then launch a real long-run training job on `data/unitree_lafan_29dof`.

**Core code changes:**
- `src/holosoma/holosoma/config_types/command.py`
  - Added multi-clip motion config fields:
    - `motion_files`
    - `motion_glob`
    - `clip_weighting_strategy`
    - adaptive clip-weight bounds / update interval
- `src/holosoma/holosoma/managers/command/terms/wbt.py`
  - Added `MotionLibrary` wrapper over multiple compatible motion clips.
  - Added per-environment `clip_ids` and batched clip-aware gathering for:
    - `joint_pos`, `joint_vel`
    - `body_pos_w`, `body_quat_w`
    - `body_lin_vel_w`, `body_ang_vel_w`
    - object tensors when present
  - Added native multi-clip reset sampling:
    - one clip sampled per env reset
    - one start timestep sampled per env reset
    - support for `uniform_step`, `uniform_clip`, `success_rate_adaptive`
  - Added adaptive clip weighting update based on episode outcomes.
  - Kept single-clip behavior backward compatible.
- `src/holosoma/holosoma/managers/observation/terms/wbt.py`
  - Updated `future_motion_targets()` to gather future reference targets from the active clip for each env.
- `src/holosoma/holosoma/managers/termination/terms/wbt.py`
  - `motion_ends()` now uses per-env active clip length instead of a single global clip length.
- `src/holosoma/holosoma/agents/ppo/ppo.py`
  - Fixed eval metric computation so motion tracking metrics work for multi-clip runs.
- `src/holosoma/holosoma/eval_metric.py`
  - Same multi-clip fix for evaluation-time motion tracking stats.

**New training presets:**
- `src/holosoma/holosoma/config_values/wbt/g1/command.py`
  - Added `g1_29dof_wbt_command_lafan_multiclip`
- `src/holosoma/holosoma/config_values/wbt/g1/randomization.py`
  - Added `g1_29dof_wbt_randomization_lafan_pretrain`
  - Disabled or softened the main motion-pretraining obstacles:
    - push randomization off
    - base COM randomization off
    - DOF bias randomization off
- `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
  - Added `g1_29dof_wbt_future_motion_lafan_multiclip`
  - Uses:
    - `simulator:mjwarp`
    - 20s episode length
    - future-motion encoder PPO
    - LAFAN replay-data multi-clip glob
    - adaptive clip weighting

**Meaningful-learning blockers identified and addressed:**
- `max_episode_length_s=10.0` in the original WBT preset was too short relative to >100s LAFAN clips.
  - Raised to `20.0` in the LAFAN multi-clip preset to match the VideoMimic scale more closely.
- `start_at_timestep_zero_prob=0.2` and `freeze_at_timestep_zero_prob=0.95` over-focused the very first frame of each clip.
  - Set both to `0.0` for LAFAN pretraining.
- Pushes / COM randomization / DOF bias randomization were likely hurting pure motion imitation pretraining.
  - Disabled in the dedicated randomization preset.

**Reference comparison with VideoMimic:**
- VideoMimic’s native multi-clip loader (`ReplayDataLoader`) samples a clip and start index per env and supports adaptive clip weighting.
- Its standard scale is much larger:
  - `4096 envs`
  - `24 steps/env/iter`
  - `100000 iter`
  - roughly `9.83e9` env-steps
- Holosoma now matches the important structural part:
  - native multi-clip per-env sampling inside one policy run
  - adaptive clip weighting

**Validation runs completed:**
- Multi-clip smoke over `lafan_walk_and_dance`:
  - run dir: `logs/WholeBodyTracking/20260313_multiclip_smoke`
  - confirmed native multi-clip reset, PPO rollout, reward, termination, observation, checkpoint saving
- Full-preset smoke over all `lafan_replay_data` clips:
  - run dir: `logs/WholeBodyTracking/20260313_multiclip_full_smoke2`
  - `256 envs`, `1 iter`
  - eval metrics at iter 0:
    - `motion_tracking_global = 4.0294`
    - `motion_tracking_local = 4.0245`

**IsaacSim re-check on this server:**
- Retested with the correct runtime path:
  - Holosoma source tree from `holosoma_yoonsang`
  - IsaacLab site-packages from `env_isaaclab`
  - wrapper: `scripts/run_train_agent_with_isaaclab_site.py`
- The same infrastructure issues remained:
  - `libGLU.so.1` missing in the Omniverse / neuray path
  - repeated `Skipping NVIDIA GPU due CUDA being in bad state`
- Result:
  - IsaacSim can initialize far enough to build some assets, but this server is still not trustworthy for long WBT training.
  - Full training was therefore launched on the validated `mjwarp` backend instead.

**Full training launched:**
- Run name: `0313_full_training`
- W&B run id: `dzqyrz9b`
- Backend: `simulator:mjwarp`
- Dataset: `data/unitree_lafan_29dof/lafan_replay_data/*.npz` (40 clips)
- Env count: `1024`
- Iterations: `100000`
- Save interval: `5000`
- Process:
  - PID: `2090503`
  - launch log: `logs/launch/0313_full_training_20260313_012447_retry.log`
  - train dir: `logs/WholeBodyTracking/20260313_0313_full_training`

**Observed early training status after launch:**
- Iteration 0:
  - `motion_tracking_global = 3.8742` (from W&B/output log)
  - `motion_tracking_local = 3.8693`
- Iteration 1:
  - `motion_tracking_global = 3.7969`
  - `motion_tracking_local = 3.7921`
- Iteration 2:
  - `motion_tracking_global = 3.8843`
  - `motion_tracking_local = 3.8800`
- Iteration 3:
  - `motion_tracking_global = 3.8846`
  - `motion_tracking_local = 3.8806`
- Throughput:
  - about `2360 steps/s`
  - `1024 envs x 24 steps = 24576 timesteps/iter`

This is the first Holosoma run in this repo that performs native multi-clip LAFAN motion tracking inside a single PPO policy update loop, instead of external checkpoint chaining.

---

### 7. Tuned Multi-Clip Pretraining Relaunch

**Problem found after the first long run:**
- The first `0313_full_training` launch was structurally correct but not a good pretraining preset.
- Two blockers stood out:
  - PPO LR was still on Holosoma default scale (`1e-5` actor / critic), far below VideoMimic's imitation scale.
  - Episodes were ending too quickly under aggressive reset noise plus strict `bad_tracking` thresholds.

**Tuning applied to the existing LAFAN multi-clip preset:**
- `src/holosoma/holosoma/config_values/wbt/g1/command.py`
  - added `lafan_init_pose_config`
  - reduced reset noise for LAFAN pretraining
- `src/holosoma/holosoma/config_values/wbt/g1/termination.py`
  - added `g1_29dof_wbt_termination_lafan_pretrain`
  - relaxed:
    - `bad_ref_pos_threshold: 0.5 -> 1.0`
    - `bad_ref_ori_threshold: 0.8 -> 1.2`
    - `bad_motion_body_pos_threshold: 0.25 -> 0.5`
- `src/holosoma/holosoma/config_values/termination.py`
  - registered `g1_29dof_wbt_lafan_pretrain`
- `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
  - updated `g1_29dof_wbt_future_motion_lafan_multiclip` to a more videomimic-like PPO scale:
    - `num_learning_iterations=100000`
    - `save_interval=500`
    - `entropy_coef=0.0025`
    - `desired_kl=0.02`
    - `num_learning_epochs=5`
    - `actor_learning_rate=critic_learning_rate=1e-3`
    - `init_noise_std=0.5`
    - `termination=g1_29dof_wbt_termination_lafan_pretrain`

**Smoke validation after tuning:**
- Run dir: `logs/WholeBodyTracking/20260313_multiclip_tuned_smoke`
- Backend: `mjwarp`
- `256 envs`, `3 iter`
- Results:
  - Iter 0:
    - `motion_tracking_global = 3.5127`
    - `motion_tracking_local = 3.4988`
    - `Mean episode length = 17.86`
  - Iter 1:
    - `motion_tracking_global = 3.5580`
    - `motion_tracking_local = 3.5464`
    - `Mean episode length = 26.39`
  - Iter 2:
    - `motion_tracking_global = 3.7219`
    - `motion_tracking_local = 3.7087`
    - `Mean episode length = 24.23`
- Key improvement:
  - early collapse improved from ~4 steps to ~18-26 steps during smoke.

**Clean full run relaunched:**
- Archived the mixed first attempt:
  - `logs/WholeBodyTracking/20260313_0313_full_training_attempt1_mixed`
- Active clean run:
  - Run name: `0313_full_training`
  - W&B account: `5yoondori-seoul-national-university`
  - W&B run id: `2n3se4x0`
  - Backend: `simulator:mjwarp`
  - Dataset: `data/unitree_lafan_29dof/lafan_replay_data/*.npz`
  - Env count: `1024`
  - Iterations: `100000`
  - PID: `2101788`
  - GPU: `1`
  - Launch log: `logs/launch/0313_full_training_20260313_tuned_clean.log`
  - Train dir: `logs/WholeBodyTracking/20260313_0313_full_training`
- Early clean-run metrics:
  - Iter 0:
    - `motion_tracking_global = 3.4642`
    - `motion_tracking_local = 3.4512`
  - Iter 1:
    - `motion_tracking_global = 3.5654`
    - `motion_tracking_local = 3.5527`

---

## 2026-02-21

### Goal
Run motion-tracking training in Holosoma using LAFAN mocap data as a pretraining step before fine-tuning on Real2Sim stairs data.

---

### 1. VideoMimic Pretrained Policy — Incompatibility Analysis

**File investigated:** `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/logs/g1_deepmimic/20250410_063030_g1_deepmimic/model_300000.pt`

**Conclusion: Cannot use as initialization for Holosoma fine-tuning.**

| | VideoMimic | Holosoma WBT |
|---|---|---|
| DOF | 23 (wrists locked) | 29 |
| Actor input dim | 415 | 154 |
| Actor output dim | 23 | 29 |
| Actor key format | `actor.0.weight` | `actor_module.module[0].weight` |
| Checkpoint format | `model_state_dict` (single) | `actor_model_state_dict` + `critic_model_state_dict` |

The observation spaces are completely different; there is no meaningful way to map weights between the two.

---

### 2. LAFAN Data Investigation

**Location:** `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/unitree_lafan/`
- `lafan_replay_data/` — 40 clips
- `lafan_walk_and_dance/` — 17 clips

**Format (VideoMimic LAFAN pkl):**
- 23-DOF joints (no wrist joints)
- Quaternion convention: xyzw (w last)
- FPS: 30
- Root z ≈ 0.785 m (correct for G1)
- 39 link names

**Verdict:** 23-DOF only — no 29-DOF version exists. Wrist joints (6 total: left/right × roll/pitch/yaw) are absent and must be zero-padded for Holosoma.

---

### 3. Conversion Script: LAFAN pkl → Holosoma npz

**Created:** `scripts/convert_lafan_pkl_to_holosoma_npz.py`

**What it does:**
- Loads VideoMimic LAFAN pkl files (23-DOF, 30 FPS, xyzw quaternion)
- Zero-pads 6 missing wrist joints → 29-DOF
- Converts quaternion convention: xyzw → wxyz
- Upsamples 30 FPS → 50 FPS (linear interp for positions/joints, SLERP for quaternions)
- Runs MuJoCo FK using `g1_29dof.xml` (nbody=51) to compute body positions/orientations
- Computes numerical velocities (joint_vel, body_lin_vel_w, body_ang_vel_w)
- Saves Holosoma-compatible npz: `joint_pos(T,36)`, `joint_vel(T,35)`, `body_pos_w(T,51,3)`, `body_quat_w(T,51,4)`, etc.

**Run with:** `holosoma_yoonsang` conda env (has mujoco)

**3 clips successfully converted → `data/lafan_pretrain/`:**
| File | Frames | FPS |
|---|---|---|
| `env_10_walk3_subject3.npz` | 2500 | 50 |
| `env_11_dance1_subject2.npz` | 2500 | 50 |
| `env_12_dance2_subject5.npz` | 2500 | 50 |

---

### 4. Training Environment

**Using:** IsaacSim (`hssim`) — already set up at `~/.holosoma_deps/miniconda3/envs/hssim/`
- Python 3.11, PyTorch 2.7.0+cu128
- 8× RTX 3090 GPUs
- `holosoma` installed, `wandb 0.24.1`

---

### 5. Errors Encountered and Fixed

#### Error 1: `CUDA_VISIBLE_DEVICES` conflicts with IsaacSim
- **Symptom:** "Setting CUDA_VISIBLE_DEVICES can lead to undesired behavior", "Skipping NVIDIA GPU due CUDA being in bad state"
- **Cause:** IsaacSim uses Vulkan/Omniverse for GPU enumeration, which conflicts with `CUDA_VISIBLE_DEVICES`
- **Fix:** Removed `CUDA_VISIBLE_DEVICES=0` from the launch command; IsaacSim selects GPU 0 automatically via `physics_gpu=0, active_gpu=0`

#### Error 2: Video encoding failed — "Video creation failed"
- **Symptom:** `RuntimeError: Video encoding failed: Video creation failed`
- **Cause:** `output_format="h264"` requires `ffmpeg` on PATH; `ffmpeg` is installed at `~/.holosoma_deps/miniconda3/envs/hssim/bin/ffmpeg` but the conda env was not activated when running the Python binary directly
- **Fix:** Added `PATH="$HOME/.holosoma_deps/miniconda3/envs/hssim/bin:$PATH"` to the launch command

---

### 6. Training Launch (3rd Attempt — Currently Running)

**WandB run:** https://wandb.ai/5yoondori-seoul-national-university/WholeBodyTracking/runs/hz5jz1t9

**Log file:** `/tmp/holosoma_train3.log`

**PID:** 3192637

**Launch command:**
```bash
PATH="$HOME/.holosoma_deps/miniconda3/envs/hssim/bin:$PATH" \
WANDB_API_KEY="54d5951df3502e196e3a1b895e227e9969fc8e7b" \
OMNI_KIT_ACCEPT_EULA=1 \
~/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --training.num-envs 2048 \
    --algo.config.num-learning-iterations 1000 \
    --algo.config.save-interval 100 \
    --command.setup-terms.motion-command.params.motion-config.motion-file="/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/lafan_pretrain/env_10_walk3_subject3.npz" \
    --logger.video.enabled True \
    --logger.video.interval 2000 \
    --logger.headless-recording True
```

**Key settings:**
- Preset: `g1-29dof-wbt` (Whole Body Tracking, flat terrain)
- Num envs: 2048 (reduced from default 8192 for test)
- Iterations: 1000 (reduced from default 40000)
- Motion file: `env_10_walk3_subject3.npz` (walk motion)
- Video: enabled, 640×360, h264

---

### 7. Known Issue — Video Interval

**Problem:** `--logger.video.interval 2000` is set in *episodes* (not iterations). This is likely too infrequent and may produce 0 videos over 1000 iterations.

**Calculation for correct interval:**
- Max episode length: 10.0 s × 50 fps = 500 steps
- PPO rollout: 24 steps per iteration
- Episodes per env per 100 iterations: (100 × 24) / 500 ≈ 4.8
- **Target interval: ~5 episodes** to get 1 video per 100 iterations (10 total)

**Next step:** After current run finishes (or is killed), relaunch with `--logger.video.interval 5`.

---

### 8. Next Steps

1. Kill current training if `video.interval=2000` will produce no videos
2. Relaunch with `--logger.video.interval 5` to achieve ~10 videos over 1000 iterations
3. Wait for training to complete (~60 min)
4. Verify video uploads to WandB
5. After training completes: detailed explanation of simulator, hyperparameters, and config
6. Fine-tune on stairs data under `data/stairs/`

---

## 2026-02-23

### Goal
Run BeyondMimic (Whole Body Tracking) training on the stairs scene and investigate WandB logging issues discovered during the run.

---

### 1. BeyondMimic Training Run

Launched a 2000-iteration WBT training run on the stairs scene using the `exp:g1-29dof-wbt` + `terrain:terrain-load-obj` presets.

**Data:**
- Motion reference: `data/stairs/motion_for_holosoma.npz` (465 frames, 50 fps, ~9.3 s stair-climbing)
- Scene mesh: `data/stairs/scene_mesh_gravity_aligned.obj` (static trimesh, z-up)

**Key config:**
- 2048 parallel envs, all sharing the same stairs scene (`env_spacing=0.0`)
- 2000 iterations, checkpoints every 200 iterations
- Video logging: `--logger.video.interval 5` (episodes) ≈ 1 video per 2–3 training iterations
- `CUDA_VISIBLE_DEVICES=0` kept per request (produces "CUDA bad state" warnings from IsaacSim's Vulkan enumeration, but training runs correctly)

**WandB run:** https://wandb.ai/5yoondori-seoul-national-university/beyondmimic/runs/qo69q8mz

**Checkpoints:** `logs/beyondmimic/20260223_g1_29dof_wbt_manager/model_00000.pt` … `model_02000.pt`

**Log file:** `/tmp/holosoma_beyondmimic.log`

**Full technical writeup:** `beyondmimic.md`

---

### 2. "Commands: Not Available" in Video Overlay — Root Cause

Every rollout video in WandB showed the text overlay "Commands: Not available". Investigation revealed this is a **cosmetic issue** unrelated to training quality.

**Root cause:**
- `simulator/shared/video_recorder.py:502` reads `getattr(self.simulator, "commands", None)` to build the overlay text
- `simulator.commands` is set only by `LocomotionManager` (locomotion training) — it holds scalar velocity commands (vx, vy, heading)
- `WholeBodyTrackingManager` never sets `simulator.commands`; its motion commands are 58-dimensional body-pose targets managed entirely inside `MotionCommand`
- `format_command_labels(None)` in `video_utils.py:89` returns the literal string `"Commands: Not available"`

**Conclusion:** Training is working correctly. The overlay simply has no way to display WBT body-pose commands, which are not a small set of scalars.

---

### 3. WandB Metric Guide — Key Metrics and Units

**`Env/average_episode_length`:**
- Unit: **control steps at 50 Hz** (1 step = 0.02 s)
- Maximum: 500 steps = 10.0 s
- At ~1000 iterations: value was ~12.4 steps (≈ 0.25 s), increasing slowly from ~11.9 at 300 iterations
- Short episodes indicate the `bad_tracking` termination is firing immediately on nearly every reset

**Three termination conditions (from `config_values/wbt/g1/termination.py`):**
| Condition | Threshold | Meaning |
|---|---|---|
| `timeout` | 500 steps | Normal end — good termination |
| `motion_ends` | clip end | Motion clip finished — good |
| `bad_tracking` | pelvis pos >0.5 m, gravity dir >0.8, ankle/wrist pos >0.25 m | Policy drifted — bad |

**Most important metrics to watch:**
| Metric | Unit | Good sign |
|---|---|---|
| `Env/average_episode_length` | steps | Increasing toward 465 |
| `Env/motion/error_ref_pos` | meters | Decreasing toward 0 |
| `Env/motion/error_body_pos` | meters | Decreasing toward 0 |
| `Env/motion/error_joint_pos` | radians | Decreasing toward 0 |
| `Train/mean_reward` | dimensionless | Increasing |
| `Policy/mean_noise_std` | — | Decreasing from 1.0 |

---

### 4. WandB Logging Gap — Bug Investigation and Fix

**Symptom:** WandB showed no metric data for training steps 13–283. Steps 0–12 and 284+ were visible. Confirmed via CSV export (not a UI rendering artifact).

**Root cause confirmed** from WandB debug log (`/tmp/wandb/run-20260223_055318-qo69q8mz/logs/debug-internal.log`):

```json
{"msg":"handler: ignoring partial history record","step":0,"current":1}
{"msg":"handler: ignoring partial history record","step":11,"current":12}
{"msg":"handler: ignoring partial history record","step":14,"current":15}
... (continuous from step 14 onward)
```

**Cause:** Two `wandb.log` call sites used conflicting step semantics in the same run:
- **PPO metrics** (`logging_utils.py:342`): `wandb.log(data, step=it)` — **explicit step**
- **Video upload** (`video_utils.py:211`): `wandb.log({"Training rollout": wandb.Video(...)})` — **no step (implicit auto-increment)**

WandB's internal step counter advances whenever an implicit `wandb.log()` call is made. When a video was uploaded inside `_rollout_step` (before PPO's `_post_epoch_logging`), the implicit log advanced the counter to N+1. The subsequent PPO call with `step=N` arrived "in the past" and was silently dropped as a "partial history record". This cascaded continuously from step 14 onward because by that point video uploads were happening at nearly every iteration.

**Fix — 4 files modified:**

1. **`src/holosoma/holosoma/utils/video_utils.py`** — `create_video()` now accepts a `step=None` parameter; passes it to `wandb.log` when set.

2. **`src/holosoma/holosoma/simulator/shared/video_recorder.py`** — Added `_wandb_step: int | None = None` field and `set_wandb_step(step)` method to `VideoRecorderInterface`; `_encode_and_save_video()` passes `step=self._wandb_step` to `create_video()`.

3. **`src/holosoma/holosoma/simulator/base_simulator/base_simulator.py`** — Added `set_video_step(step)` public method that calls `self.video_recorder.set_wandb_step(step)` when a recorder is configured.

4. **`src/holosoma/holosoma/agents/ppo/ppo.py`** — At the top of each training iteration (before `_rollout_step`), PPO now calls `simulator.set_video_step(it)` via `getattr(self.env, "simulator", None)`.

With this fix, every video `wandb.log` uses `step=it` matching the PPO metrics log for the same iteration, preventing the step counter race condition.

---

## 2026.02.24

### Goal
Validate custom Real2Sim scene/motion data in Isaac (not only MuJoCo), first with replay and then with an already-trained policy, while keeping the existing `holosoma` setup intact.

---

### 1. Repository / Branch Setup

- Kept the existing working branch/environment unchanged.
- Added an additional local branch for the updated server code path:
  - Local branch: `kyungminlee_server`
  - Current pointer: commit `6d8d895` (same content as `kyungminn_server` in this clone)
- Confirmed the existing branch remains available (`main`, `yoonsang`, etc.).

---

### 2. Data + Visualization Outputs Organization

- Used the existing `data/<video>/` layout with:
  - `scene_mesh_gravity_aligned.obj`
  - `motion_for_holosoma.npz` (29-DoF G1)
- Consolidated generated videos under this repository for VS Code visibility:
  - `tmp/holosoma_previews/` (MuJoCo preview videos)
  - `tmp/isaac_previews/` (Isaac replay videos)
  - `tmp/isaac_policy_eval*/` (Isaac policy-eval videos)

---

### 3. Isaac Evaluation Paths Investigated

#### A) Replay path (reference motion only)

- Entry point used: `src/holosoma/holosoma/replay.py`
- Purpose: verify scene/motion alignment and obvious collision/setup issues.
- Outputs: `tmp/isaac_previews/{handstand,ladder2,pg_b1,stairs,wall-kicking}/episode_0_*.mp4`

#### B) DDS-based policy path (attempted, then avoided)

- Attempted through `holosoma_inference` policy runtime.
- Encountered CycloneDDS interface/domain creation failures on this server:
  - `failed to enumerate interfaces for "udp": -1`
  - `Failed to create domain explicitly`
  - Logs: `/tmp/policy_smoketest.log`, `/tmp/policy_smoketest_eno1.log`
- Because of this, we switched to non-DDS evaluation for Isaac policy rollout.

#### C) Non-DDS policy evaluation path (final chosen path)

- Entry point: `src/holosoma/holosoma/eval_agent.py`
- This runs policy + Isaac env in one process (no DDS bridge).
- Confirmation from logs: `Robot bridge disabled` from `base_simulator._init_bridge`.

---

### 4. Issues Encountered in Isaac and How Each Was Resolved

| Issue | Symptom | Root Cause | Resolution |
|---|---|---|---|
| DDS network/domain init failure | CycloneDDS runtime error when trying `holosoma_inference` | Server/network interface not usable for DDS in this context | Switched to non-DDS `eval_agent.py` path for policy evaluation |
| Optional import fragility in inference runtime | `netifaces` / `sshkeyboard` dependency sensitivity | Hard imports in `holosoma_inference` base policy | Made imports optional + graceful handling in `src/holosoma_inference/holosoma_inference/policies/base.py` |
| USD-related confusion | Repeated log: `No USD scene objects found` | Scenes were loaded as OBJ terrain meshes, not USD scene assets | No code fix required; treated as expected informational log for this dataset format |
| PhysX mesh cooking failures | `UjitsoMeshCookingContext: cooking failure` and `Unable to create triangle mesh for /World/ground/mesh` (multiple datasets) | `load_obj` terrain tiled by default (`num_rows=10`, `num_cols=20`) -> huge triangle count for large scenes | For eval/validation runs, forced `num_rows=1`, `num_cols=1` to avoid tiling explosion |
| Severe GPU/physics crash during replay | Warp OOM/illegal memory access, then `Failed to get DOF velocities from backend` in stairs replay | Same large tiled-mesh pressure path as above | Same mitigation (`num_rows=1`, `num_cols=1`) removed mesh-cooking failures in follow-up runs |
| Replay video speed too fast | Isaac replay looked significantly faster than MuJoCo/real motion | Frame capture was called at control rate, but recorder still applied control-step decimation (`control_decimation=4`) | Added `control_step=True` bypass to recorder capture path; replay now captures every control step |
| Headless replay debug marker edge case | Potential marker access issue in headless replay path | Debug visualization objects may be absent in some replay flows | Added guard in `_draw_debug_vis_isaacsim` to skip drawing when markers are unavailable |
| Headless Isaac shutdown hang | Process often exits via timeout (`124`) even after video creation | Known teardown behavior in this environment | Used `timeout` operationally; treated outputs as valid when videos were already written |

---

### 5. Code Changes Made on Existing Files (2026.02.24 workstream)

1. `src/holosoma/holosoma/simulator/shared/video_recorder.py`
- `capture_frame()` now supports `control_step: bool = False`.
- When `control_step=True`, decimation is bypassed so replay/control-rate callers do not under-sample frames.

2. `src/holosoma/holosoma/simulator/base_simulator/base_simulator.py`
- `capture_video_frame()` now forwards `control_step` to the recorder.

3. `src/holosoma/holosoma/envs/wbt/wbt_manager.py`
- Replay path now calls `capture_video_frame(env_id=0, control_step=True)`.
- Added safety guard in `_draw_debug_vis_isaacsim()` when visualization markers are not initialized.

4. `src/holosoma/holosoma/managers/terrain/terms/locomotion.py`
- In deterministic eval spawn (`randomize_tiles=False`), `_get_env_origins()` now supports OBJ terrain origin grids by using `_get_load_obj_env_origin_grid()` when `_env_origins` is absent.

5. `src/holosoma_inference/holosoma_inference/policies/base.py`
- Made `netifaces` and `sshkeyboard` optional imports.
- Added explicit error message for booster mode if `netifaces` is missing.
- Keyboard listener now degrades gracefully if `sshkeyboard` is unavailable.

---

### 6. Isaac Runs Performed (Policy Eval, Non-DDS)

- Checkpoint used: `logs/beyondmimic/20260223_g1_29dof_wbt_manager/model_02999.pt`
- Datasets: `handstand`, `ladder2`, `pg_b1`, `stairs`, `wall-kicking`

Initial policy-eval logs (before tiling fix):
- `/tmp/isaac_policy_eval_handstand.log`
- `/tmp/isaac_policy_eval_ladder2.log`
- `/tmp/isaac_policy_eval_pg_b1.log`
- `/tmp/isaac_policy_eval_stairs.log`
- `/tmp/isaac_policy_eval_wall-kicking.log`

Fixed policy-eval logs (`num_rows=1`, `num_cols=1`):
- `/tmp/isaac_policy_eval_fix_handstand.log`
- `/tmp/isaac_policy_eval_fix_ladder2.log`
- `/tmp/isaac_policy_eval_fix_pg_b1.log`
- `/tmp/isaac_policy_eval_fix_stairs.log`
- `/tmp/isaac_policy_eval_fix_wall-kicking.log`

Final video outputs for fixed runs:
- `tmp/isaac_policy_eval_fix/handstand/`
- `tmp/isaac_policy_eval_fix/ladder2/`
- `tmp/isaac_policy_eval_fix/pg_b1/`
- `tmp/isaac_policy_eval_fix/stairs/`
- `tmp/isaac_policy_eval_fix/wall-kicking/`

Collision warning summary for fixed runs:
- `cook_fail=0`, `tri_fail=0`, `gpu_err=0` for all five datasets.

---

### 7. Replay Speed Fix Validation

- Validation log: `/tmp/isaac_replay_speedcheck_handstand.log`
- Key indicator after fix: `Frame capture stats: n=682` (instead of ~1/4 frame count previously).
- Output video: `tmp/isaac_replay_speedcheck/handstand/episode_0_1771931510.mp4`

---

## 2026.02.26

### Branch Reset / Restart

- Switched working branch to `kyungminn_server` with forced checkout:
  - Command: `git checkout -f kyungminn_server`
  - Verified current branch: `kyungminn_server`
- Preserved requested newly created artifacts:
  - `progress.md`
  - `tmp/`
  - `data/` (including custom scene/motion files)

### Replay Consistency Recheck (kyungminn_server)

- Replay entrypoint tested: `src/holosoma/holosoma/replay.py`
- Discovered branch issue in headless replay:
  - `AttributeError: 'MotionCommand' object has no attribute 'visualization_markers'`
  - Cause: replay always calls `_draw_debug_vis_isaacsim()`, but markers are initialized only when viewer exists.
- Applied minimal guard to support headless replay:
  - File: `src/holosoma/holosoma/envs/wbt/wbt_manager.py`
  - Change: return early in `_draw_debug_vis_isaacsim()` if `visualization_markers` is absent.
- Recheck runs (all finished without traceback; close hook reached):
  - Logs:
    - `/tmp/kyungminn_recheck_handstand_20260226.log`
    - `/tmp/kyungminn_recheck_ladder2_20260226.log`
    - `/tmp/kyungminn_recheck_pg_b1_20260226.log`
    - `/tmp/kyungminn_recheck_stairs_20260226.log`
    - `/tmp/kyungminn_recheck_wall-kicking_20260226.log`
  - Each log contains `Successfully patched close_stage method` and no Python traceback.

---

## 2026.02.28

### Critical Issue: IsaacSim Reset-Sync Bug (immediate episode collapse)

#### Symptom
- In IsaacSim WBT runs, episodes frequently terminated at length 1.
- After `motion_command.reset()` selected a valid motion frame, robot state snapped back to default origin/pose during reset refresh.
- This caused large tracking error on the first step and immediate `bad_tracking` termination.

#### Root Cause (two coupled bugs)
1. **Root-state proxy sync hazard**
   - `MotionCommand.reset()` wrote robot root state using partial slice assignment.
   - With IsaacSim `RootStatesProxy`, partial slice writes can desynchronize the xyzw view from the backing wxyz buffer used for simulator writes.

2. **IsaacLab write path mismatch with indexed writes**
   - `IsaacSim.set_actor_root_state_tensor_robots()` and `IsaacSim.set_dof_state_tensor_robots()` used indexed writes (`env_ids`, `joint_ids`).
   - In this setup, those indexed articulation writes did not correctly propagate target reset states.
   - Result: reset refresh read back near-default root/DOF states.

#### Fix Applied
- `src/holosoma/holosoma/managers/command/terms/wbt.py`
  - Replaced partial root slice writes with full-row assignment when writing reset root state (keeps proxy xyzw/wxyz consistent).
  - Applied same full-row pattern in default-pose interpolation body-state capture helper.

- `src/holosoma/holosoma/simulator/isaacsim/isaacsim.py`
  - Reworked robot root reset write:
    - Build full root-state tensor (`_robot.data.root_state_w`) and write all envs in one call (no indexed env write).
  - Reworked DOF reset write:
    - Build full articulation-order joint tensors from desired DOF states via `dof_ids` mapping.
    - Write all joints/envs in one call (no indexed `joint_ids`/`env_ids` write).

#### Sanity Check (post-fix)
- Reset/refresh diagnostic (`stairs`, IsaacSim):
  - `/tmp/diag_after_fix.log`
  - Robot root after refresh stays aligned with motion target (no snap to origin).
- Short rollout sanity (`stairs`, 1 env):
  - `/tmp/diag_sanity_step.log`
  - 5 consecutive steps with `done=[0]` (no immediate termination).
- Tiny training run (3 iterations):
  - `/tmp/train_sanity_20260228.log`
  - Training completed normally; sample log at iter 2: `Mean episode length: 5.92` (not collapsed to 1-step resets).

---

## 2026-03-13

### IsaacSim Multi-Clip WBT Recovery

- Goal:
  - Recover IsaacSim training for the native Holosoma multi-clip LAFAN motion-tracking setup.
  - Match the current MuJoCo run structurally:
    - `exp:g1-29dof-wbt-future-motion-lafan-multiclip`
    - single policy over all 40 `lafan_replay_data` clips
    - PPO future-motion encoder
    - W&B logging enabled

### Root Causes Confirmed

- IsaacSim itself was available from the `holosoma_yoonsang` runtime when paired with IsaacLab site-packages from `env_isaaclab`.
- Two environment/runtime blockers were identified:
  - URDF->USD conversion failed when the cache target lived on the NAS-backed workspace path.
  - `libGLU.so.1` was missing for IsaacSim's RTX / neuray dependency chain.
- A third startup drag was self-inflicted:
  - robot prim tree + full robot property dumps were being emitted on every IsaacSim startup.

### Fixes Applied

- `src/holosoma/holosoma/simulator/isaacsim/isaacsim.py`
  - changed default USD conversion cache root to local tmp storage via `tempfile.gettempdir()`
  - left `HOLOSOMA_USD_CONVERSION_ROOT` as an explicit override
  - gated expensive prim/property debug dumps behind `HOLOSOMA_ISAAC_DEBUG_PRIMS=1`

- `scripts/run_train_agent_with_isaaclab_site.py`
  - when late-loading IsaacLab site-packages, also prepend the inferred Isaac env `lib` / `lib64` dirs to `LD_LIBRARY_PATH`

- Runtime package fix:
  - installed `libglu` into `env_isaaclab` from `conda-forge`

### Validation

- IsaacSim smoke without W&B:
  - run dir: `logs/WholeBodyTracking/20260313_isaac_multiclip_tuned_smoke5`
  - result: full one-iteration PPO run completed, checkpoint written
  - eval metric at iter 0:
    - `motion_tracking_global = 0.0953 m`
    - `motion_tracking_local = 0.0896 m`

- IsaacSim smoke with W&B:
  - run dir: `logs/WholeBodyTracking/20260313_isaac_multiclip_wandb_smoke`
  - W&B run: `8ltnr32i`
  - result: full one-iteration PPO run completed with online logging
  - eval metric at iter 0:
    - `motion_tracking_global = 0.0597 m`
    - `motion_tracking_local = 0.0619 m`

### Long IsaacSim Training Run Started

- Active run:
  - name: `0313_full_training_isaac`
  - W&B run: `1clldux7`
  - URL: `https://wandb.ai/5yoondori-seoul-national-university/WholeBodyTracking/runs/1clldux7`
  - runtime session: PTY session `70953`
  - simulator: `isaacsim`
  - device: `cuda:0` (physical GPU 0)
  - envs: `256`
  - iterations: `100000`
  - dataset: `data/unitree_lafan_29dof/lafan_replay_data/*.npz` (40 clips)

- Observed early status:
  - iteration 0:
    - `motion_tracking_global = 0.1244 m`
    - `motion_tracking_local = 0.1075 m`
    - `Mean episode length = 17.71`
  - iteration 2:
    - `motion_tracking_global = 0.1202 m`
    - `motion_tracking_local = 0.1050 m`
    - `Mean episode length = 25.89`
  - iteration 3:
    - `motion_tracking_global = 0.1178 m`
    - `motion_tracking_local = 0.1041 m`
    - `Mean episode length = 21.93`

### Operational Notes

- On this GPU server, `--headless` should stay enabled for training.
- IsaacSim still enumerates all GPUs and leaves small contexts on each, even when compute is bound to `--device cuda:0`.
- The current long run is stable through multiple iterations and is the first validated IsaacSim multi-clip WBT training run for the LAFAN 29DOF set on this machine.

---

## 2026-03-16

### Holosoma LAFAN Preset Realignment Toward videomimic Stage-1

- Goal:
  - Add a dedicated Holosoma preset that is structurally closer to videomimic stage-1 pretraining for multi-clip LAFAN motion tracking.
  - Preserve all existing Holosoma WBT presets and workflows unchanged.

### Why This Change Was Needed

- The earlier Holosoma LAFAN setup differed from videomimic stage-1 in four important ways:
  - actor input did not directly include torso XY / yaw tracking errors or target joint targets,
  - reward did not include explicit joint position / velocity tracking terms,
  - termination was looser and only checked a small subset of tracked bodies,
  - action scale was much larger (`1.0` vs videomimic's `0.25`), making each policy action move joint targets more aggressively.

### Files Updated

- `src/holosoma/holosoma/managers/observation/terms/wbt.py`
  - added stage-1 style observation terms:
    - `torso_xy_rel`
    - `torso_yaw_rel`
    - `target_joint_pos`
    - `target_root_roll_pitch`

- `src/holosoma/holosoma/managers/reward/terms/wbt.py`
  - added explicit imitation rewards:
    - `motion_joint_position_error_exp`
    - `motion_joint_velocity_error_exp`

- `src/holosoma/holosoma/managers/termination/terms/wbt.py`
  - added `TrackedBodyPositionErrorThreshold`
  - this mirrors videomimic stage-1 style termination more closely by ending the episode when any tracked body deviates too far from the reference after the first few steps

- `src/holosoma/holosoma/config_values/wbt/g1/observation.py`
  - added `g1_29dof_wbt_observation_lafan_videomimic_stage1`
  - actor input is now split into:
    - `actor_state_history` (5-step history of ang vel, gravity, dof pos/vel, actions)
    - `actor_tracking_history` (5-step history of torso XY / yaw tracking errors)
    - `actor_targets` (target joint pose and target root roll/pitch)

- `src/holosoma/holosoma/config_values/wbt/g1/reward.py`
  - added `g1_29dof_wbt_reward_lafan_videomimic_stage1`
  - reward weights/sigmas were matched to videomimic stage-1 style terms as closely as Holosoma's current reward API allows

- `src/holosoma/holosoma/config_values/wbt/g1/termination.py`
  - added `g1_29dof_wbt_termination_lafan_videomimic_stage1`
  - this uses tracked-body cartesian error thresholding instead of the older, looser `bad_tracking` preset

- `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
  - added `g1_29dof_wbt_lafan_videomimic_stage1`
  - key settings:
    - `num_envs = 4096`
    - `num_learning_iterations = 100000`
    - PPO MLP widths `[1024, 512, 256, 128]`
    - `action_scale = 0.25`
    - `num_learning_epochs = 5`
    - `num_mini_batches = 4`
    - `actor/critic lr = 1e-3`
    - `desired_kl = 0.02`

- `src/holosoma/holosoma/config_values/observation.py`
- `src/holosoma/holosoma/config_values/reward.py`
- `src/holosoma/holosoma/config_values/termination.py`
- `src/holosoma/holosoma/config_values/experiment.py`
  - registered the new preset in the top-level default registries for CLI access

### Compatibility / Safety

- Existing Holosoma presets were not overwritten.
- The new videomimic-aligned setup is opt-in via:
  - `exp:g1-29dof-wbt-lafan-videomimic-stage1`
- This means future motion-tracking experiments do not need to use the videomimic-style setup unless explicitly requested.

### Validation

- Import validation passed in `holosoma_yoonsang`:
  - new experiment preset resolves correctly
  - actor input keys resolve to:
    - `actor_state_history`
    - `actor_tracking_history`
    - `actor_targets`
  - critic input keys resolve to:
    - `critic_obs`
    - `actor_state_history`
    - `actor_tracking_history`
    - `actor_targets`

- Python bytecode compilation passed for all touched files.
