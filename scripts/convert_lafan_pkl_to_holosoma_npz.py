#!/usr/bin/env python3
"""Convert VideoMimic LAFAN pkl files (23-DOF) to Holosoma npz format (29-DOF).

Reuses the MuJoCo FK and velocity computation from convert_h5_to_holosoma_npz.py.

Usage:
    python scripts/convert_lafan_pkl_to_holosoma_npz.py \
        --input-dir /path/to/videomimic/simulation/data/unitree_lafan/lafan_walk_and_dance \
        --output-dir data/lafan_pretrain \
        --max-clips 5
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUJOCO_MODEL_PATH = (
    PROJECT_ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
)
OUTPUT_FPS = 50
LAFAN_FPS = 30  # LAFAN1 dataset native FPS

G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# ---------------------------------------------------------------------------
# Helpers (same as convert_h5_to_holosoma_npz.py)
# ---------------------------------------------------------------------------
def pad_joints_23_to_29(
    joints_23: np.ndarray,
    pkl_joint_names: list[str],
) -> np.ndarray:
    T = joints_23.shape[0]
    joints_29 = np.zeros((T, 29), dtype=np.float64)
    for target_idx, target_name in enumerate(G1_29DOF_JOINT_NAMES):
        if target_name in pkl_joint_names:
            src_idx = pkl_joint_names.index(target_name)
            joints_29[:, target_idx] = joints_23[:, src_idx]
    return joints_29


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def interpolate_positions(data: np.ndarray, times_in: np.ndarray, times_out: np.ndarray) -> np.ndarray:
    from scipy.interpolate import interp1d
    original_shape = data.shape
    if data.ndim == 1:
        data = data[:, np.newaxis]
    T_in = data.shape[0]
    data_flat = data.reshape(T_in, -1)
    f = interp1d(times_in, data_flat, axis=0, kind="linear", fill_value="extrapolate")
    result_flat = f(times_out)
    new_shape = (len(times_out),) + original_shape[1:]
    return result_flat.reshape(new_shape)


def interpolate_quaternions(quats: np.ndarray, times_in: np.ndarray, times_out: np.ndarray) -> np.ndarray:
    quats_xyzw = np.concatenate([quats[:, 1:4], quats[:, 0:1]], axis=-1)
    rots = Rotation.from_quat(quats_xyzw)
    slerp = Slerp(times_in, rots)
    times_clamped = np.clip(times_out, times_in[0], times_in[-1])
    rots_out = slerp(times_clamped)
    quats_out_xyzw = rots_out.as_quat()
    quats_out_wxyz = np.concatenate([quats_out_xyzw[:, 3:4], quats_out_xyzw[:, 0:3]], axis=-1)
    return quats_out_wxyz


def mujoco_forward_kinematics(
    model: mujoco.MjModel,
    qpos_sequence: np.ndarray,
    dt: float,
) -> dict:
    data = mujoco.MjData(model)
    T = qpos_sequence.shape[0]
    nbody = model.nbody

    body_pos_w = np.zeros((T, nbody, 3), dtype=np.float64)
    body_quat_w = np.zeros((T, nbody, 4), dtype=np.float64)

    for t in range(T):
        data.qpos[:] = qpos_sequence[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos_w[t] = data.xpos[:].copy()
        body_quat_w[t] = data.xquat[:].copy()

    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
    body_ang_vel_w = _body_quat_to_angular_velocity_batch(body_quat_w, dt)

    return {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def _body_quat_to_angular_velocity_batch(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    T, N, _ = quats_wxyz.shape
    angvel = np.zeros((T, N, 3), dtype=np.float64)
    for b in range(N):
        q_xyzw = np.concatenate([quats_wxyz[:, b, 1:4], quats_wxyz[:, b, 0:1]], axis=-1)
        rots = Rotation.from_quat(q_xyzw)
        for t in range(T):
            if t == 0:
                r_diff = rots[1] * rots[0].inv()
                angvel[t, b] = r_diff.as_rotvec() / dt
            elif t == T - 1:
                r_diff = rots[T - 1] * rots[T - 2].inv()
                angvel[t, b] = r_diff.as_rotvec() / dt
            else:
                r_diff = rots[t + 1] * rots[t - 1].inv()
                angvel[t, b] = r_diff.as_rotvec() / (2.0 * dt)
    return angvel


def compute_velocities_numerical(joint_pos_full: np.ndarray, dt: float) -> np.ndarray:
    root_pos = joint_pos_full[:, :3]
    root_linvel = np.gradient(root_pos, dt, axis=0)
    root_quat_wxyz = joint_pos_full[:, 3:7]
    root_angvel = _quat_sequence_to_angular_velocity(root_quat_wxyz, dt)
    joint_angles = joint_pos_full[:, 7:]
    joint_vel = np.gradient(joint_angles, dt, axis=0)
    return np.concatenate([root_linvel, root_angvel, joint_vel], axis=1)


def _quat_sequence_to_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    T = quats_wxyz.shape[0]
    angvel = np.zeros((T, 3), dtype=np.float64)
    quats_xyzw = np.concatenate([quats_wxyz[:, 1:4], quats_wxyz[:, 0:1]], axis=-1)
    rots = Rotation.from_quat(quats_xyzw)
    for t in range(T):
        if t == 0:
            r_diff = rots[1] * rots[0].inv()
            angvel[t] = r_diff.as_rotvec() / dt
        elif t == T - 1:
            r_diff = rots[T - 1] * rots[T - 2].inv()
            angvel[t] = r_diff.as_rotvec() / dt
        else:
            r_diff = rots[t + 1] * rots[t - 1].inv()
            angvel[t] = r_diff.as_rotvec() / (2.0 * dt)
    return angvel


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert_one_pkl(
    pkl_path: Path,
    output_path: Path,
    model: mujoco.MjModel,
    max_frames: int = -1,
) -> None:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    pkl_joint_names = list(data["joint_names"])
    joints_23 = data["joints"].astype(np.float64)
    root_pos = data["root_pos"].astype(np.float64)
    root_quat_xyzw = data["root_quat"].astype(np.float64)

    T_in = joints_23.shape[0]
    if max_frames > 0 and T_in > max_frames:
        print(f"    Truncating: {T_in} → {max_frames} frames")
        joints_23 = joints_23[:max_frames]
        root_pos = root_pos[:max_frames]
        root_quat_xyzw = root_quat_xyzw[:max_frames]
        T_in = max_frames

    duration = (T_in - 1) / LAFAN_FPS
    print(f"    Input: {T_in} frames @ {LAFAN_FPS} fps = {duration:.1f}s, {joints_23.shape[1]} joints")

    # 1. Pad 23 → 29 DOF
    joints_29 = pad_joints_23_to_29(joints_23, pkl_joint_names)

    # 2. Convert quaternion xyzw → wxyz
    root_quat_wxyz = xyzw_to_wxyz(root_quat_xyzw)

    # 3. Build qpos at input FPS
    qpos_in = np.concatenate([root_pos, root_quat_wxyz, joints_29], axis=1)

    # 4. Upsample to OUTPUT_FPS
    times_in = np.arange(T_in) / LAFAN_FPS
    T_out = int(np.ceil(duration * OUTPUT_FPS)) + 1
    times_out = np.linspace(0.0, duration, T_out)
    print(f"    Upsampling: {LAFAN_FPS} → {OUTPUT_FPS} fps ({T_in} → {T_out} frames)")

    root_pos_up = interpolate_positions(root_pos, times_in, times_out)
    joints_29_up = interpolate_positions(joints_29, times_in, times_out)
    root_quat_up = interpolate_quaternions(root_quat_wxyz, times_in, times_out)
    qpos_up = np.concatenate([root_pos_up, root_quat_up, joints_29_up], axis=1)

    # 5. MuJoCo forward kinematics
    dt_out = 1.0 / OUTPUT_FPS
    print(f"    Running MuJoCo FK for {T_out} frames...")
    fk = mujoco_forward_kinematics(model, qpos_up, dt_out)

    # 6. Numerical velocities
    joint_vel = compute_velocities_numerical(qpos_up, dt_out)

    # 7. Get body/joint names from MuJoCo model
    joint_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    body_names = []
    for i in range(model.nbody):
        body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

    # 8. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.array([OUTPUT_FPS]),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        joint_pos=qpos_up,
        joint_vel=joint_vel,
        body_pos_w=fk["body_pos_w"],
        body_quat_w=fk["body_quat_w"],
        body_lin_vel_w=fk["body_lin_vel_w"],
        body_ang_vel_w=fk["body_ang_vel_w"],
    )
    print(f"    Saved: {output_path}")
    print(f"      joint_pos:  {qpos_up.shape}")
    print(f"      body_pos_w: {fk['body_pos_w'].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with LAFAN pkl files")
    parser.add_argument("--output-dir", required=True, help="Output directory for npz files")
    parser.add_argument("--max-clips", type=int, default=-1, help="Max clips to convert (-1=all)")
    parser.add_argument("--max-frames", type=int, default=3000,
                        help="Max frames per clip at 30fps (default 3000=100s). -1=no limit")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pkl_files = sorted(input_dir.glob("*.pkl"))

    if args.max_clips > 0:
        pkl_files = pkl_files[:args.max_clips]

    print(f"Found {len(pkl_files)} pkl files in {input_dir}")
    print(f"Loading MuJoCo model: {MUJOCO_MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MUJOCO_MODEL_PATH))
    print(f"  nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

    for i, pkl_path in enumerate(pkl_files):
        stem = pkl_path.stem  # e.g. "env_2_walk1_subject2"
        out_path = output_dir / f"{stem}.npz"
        print(f"\n[{i+1}/{len(pkl_files)}] {pkl_path.name} → {out_path}")
        convert_one_pkl(pkl_path, out_path, model, max_frames=args.max_frames)

    print(f"\nDone. Converted {len(pkl_files)} clips to {output_dir}/")


if __name__ == "__main__":
    main()
