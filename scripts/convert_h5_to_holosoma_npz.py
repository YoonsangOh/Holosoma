#!/usr/bin/env python3
"""Convert retargeted H5 motion files to the .npz format expected by Holosoma's WBT training.

Input (H5):
    - fps: 15 (attr)
    - joints: (T, 23) joint angles (missing 6 wrist DOFs)
    - root_pos: (T, 3)
    - root_quat: (T, 4) in xyzw convention
    - link_pos: (T, 37, 3)
    - link_quat: (T, 37, 4) in xyzw convention

Output (NPZ):
    - fps: [50]
    - joint_names: 29 G1 joint names
    - body_names: 51 MuJoCo body names
    - joint_pos: (T', 36) = [root_xyz(3), root_wxyz(4), joints_29]
    - joint_vel: (T', 35) = [root_linvel(3), root_angvel(3), joints_vel_29]
    - body_pos_w: (T', 51, 3)
    - body_quat_w: (T', 51, 4) in wxyz convention
    - body_lin_vel_w: (T', 51, 3)
    - body_ang_vel_w: (T', 51, 3)

Where T' = ceil(T * output_fps / input_fps).

Usage:
    python scripts/convert_h5_to_holosoma_npz.py          # convert all 5 folders
    python scripts/convert_h5_to_holosoma_npz.py handstand # convert one folder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
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
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FPS = 50

# The 29 joints that the G1 MuJoCo model expects (hinge joints, in order).
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
# Helpers
# ---------------------------------------------------------------------------
def read_h5(path: Path) -> dict:
    """Read an H5 motion file and return a dict of arrays + metadata."""
    with h5py.File(path, "r") as f:
        data = {
            "fps": float(f.attrs["fps"]),
            "h5_joint_names": [s for s in f.attrs["joint_names"]],
            "h5_link_names": [s for s in f.attrs["link_names"]],
            "joints": f["joints"][:].astype(np.float64),       # (T, 23)
            "root_pos": f["root_pos"][:].astype(np.float64),   # (T, 3)
            "root_quat": f["root_quat"][:].astype(np.float64), # (T, 4) xyzw
        }
    return data


def pad_joints_23_to_29(
    joints_23: np.ndarray,
    h5_joint_names: list[str],
) -> np.ndarray:
    """Map 23-DOF joint array to 29-DOF, filling missing wrist joints with 0.

    Returns (T, 29) array with joints in G1_29DOF_JOINT_NAMES order.
    """
    T = joints_23.shape[0]
    joints_29 = np.zeros((T, 29), dtype=np.float64)
    for target_idx, target_name in enumerate(G1_29DOF_JOINT_NAMES):
        if target_name in h5_joint_names:
            src_idx = h5_joint_names.index(target_name)
            joints_29[:, target_idx] = joints_23[:, src_idx]
        # else: stays 0.0 (wrist joints)
    return joints_29


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion array from xyzw to wxyz convention.

    Works for shapes (T, 4), (T, N, 4), etc.
    """
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def interpolate_positions(data: np.ndarray, times_in: np.ndarray, times_out: np.ndarray) -> np.ndarray:
    """Linear interpolation for position-like data.

    data: (..., D) at times_in
    Returns: (..., D) at times_out
    """
    from scipy.interpolate import interp1d
    # data shape: (T_in, ...) — interpolate along axis=0
    original_shape = data.shape
    if data.ndim == 1:
        data = data[:, np.newaxis]
    # Flatten all dims except time
    T_in = data.shape[0]
    data_flat = data.reshape(T_in, -1)
    f = interp1d(times_in, data_flat, axis=0, kind="linear", fill_value="extrapolate")
    result_flat = f(times_out)
    new_shape = (len(times_out),) + original_shape[1:]
    return result_flat.reshape(new_shape)


def interpolate_quaternions(quats: np.ndarray, times_in: np.ndarray, times_out: np.ndarray) -> np.ndarray:
    """Slerp interpolation for quaternion data.

    quats: (T_in, 4) in wxyz convention.
    Returns: (T_out, 4) in wxyz convention.
    """
    # scipy Rotation uses xyzw internally (scalar-last)
    quats_xyzw = np.concatenate([quats[:, 1:4], quats[:, 0:1]], axis=-1)
    rots = Rotation.from_quat(quats_xyzw)
    slerp = Slerp(times_in, rots)
    # Clamp output times to input range to avoid extrapolation
    times_clamped = np.clip(times_out, times_in[0], times_in[-1])
    rots_out = slerp(times_clamped)
    quats_out_xyzw = rots_out.as_quat()  # (T_out, 4) xyzw
    quats_out_wxyz = np.concatenate([quats_out_xyzw[:, 3:4], quats_out_xyzw[:, 0:3]], axis=-1)
    return quats_out_wxyz


def interpolate_quaternions_batch(quats: np.ndarray, times_in: np.ndarray, times_out: np.ndarray) -> np.ndarray:
    """Slerp for (T_in, N, 4) batch of quaternions in wxyz convention.

    Returns (T_out, N, 4) in wxyz.
    """
    N = quats.shape[1]
    T_out = len(times_out)
    result = np.zeros((T_out, N, 4), dtype=np.float64)
    for i in range(N):
        result[:, i, :] = interpolate_quaternions(quats[:, i, :], times_in, times_out)
    return result


def mujoco_forward_kinematics(
    model: mujoco.MjModel,
    qpos_sequence: np.ndarray,
    dt: float,
) -> dict:
    """Run MuJoCo FK for each frame and collect body states.

    qpos_sequence: (T, nq) — [root_xyz(3), root_wxyz(4), joints(29)]
    dt: timestep for numerical velocity computation
    Returns dict with body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w.
    """
    data = mujoco.MjData(model)
    T = qpos_sequence.shape[0]
    nbody = model.nbody

    body_pos_w = np.zeros((T, nbody, 3), dtype=np.float64)
    body_quat_w = np.zeros((T, nbody, 4), dtype=np.float64)  # wxyz (MuJoCo native)

    for t in range(T):
        data.qpos[:] = qpos_sequence[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        body_pos_w[t] = data.xpos[:].copy()
        body_quat_w[t] = data.xquat[:].copy()  # Already wxyz in MuJoCo

    # Compute body velocities via numerical differentiation (same approach as
    # the reference convert_data_format_mj.py which also uses post-hoc derivatives).
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
    body_ang_vel_w = _body_quat_to_angular_velocity_batch(body_quat_w, dt)

    return {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def _body_quat_to_angular_velocity_batch(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity for (T, N, 4) wxyz quaternion sequences.

    Uses central differences: omega = rotvec(q_{t+1} * q_{t-1}^{-1}) / (2*dt)
    Returns (T, N, 3).
    """
    T, N, _ = quats_wxyz.shape
    angvel = np.zeros((T, N, 3), dtype=np.float64)

    for b in range(N):
        # Convert to scipy (xyzw)
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


def compute_velocities_numerical(
    joint_pos_full: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute joint velocities via central-difference numerical differentiation.

    joint_pos_full: (T, 36) = [root_xyz, root_wxyz, joints_29]
    Returns: (T, 35) = [root_linvel(3), root_angvel(3), joints_vel(29)]
    """
    T = joint_pos_full.shape[0]

    # Root linear velocity: d(root_xyz)/dt
    root_pos = joint_pos_full[:, :3]
    root_linvel = np.gradient(root_pos, dt, axis=0)

    # Root angular velocity from quaternion derivative
    root_quat_wxyz = joint_pos_full[:, 3:7]
    root_angvel = _quat_sequence_to_angular_velocity(root_quat_wxyz, dt)

    # Joint velocities: d(joint_angles)/dt
    joint_angles = joint_pos_full[:, 7:]
    joint_vel = np.gradient(joint_angles, dt, axis=0)

    return np.concatenate([root_linvel, root_angvel, joint_vel], axis=1)


def _quat_sequence_to_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from a sequence of wxyz quaternions.

    Uses central difference: omega = 2 * quat_to_rotvec(q_{t+1} * q_{t-1}^{-1}) / (2*dt)
    Returns (T, 3) angular velocity.
    """
    T = quats_wxyz.shape[0]
    angvel = np.zeros((T, 3), dtype=np.float64)

    # Convert to scipy Rotation (xyzw)
    quats_xyzw = np.concatenate([quats_wxyz[:, 1:4], quats_wxyz[:, 0:1]], axis=-1)
    rots = Rotation.from_quat(quats_xyzw)

    # Central differences for interior, forward/backward for edges
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
def convert_one(folder_name: str, model: mujoco.MjModel) -> None:
    """Convert one video folder's H5 + OBJ to Holosoma .npz format."""
    folder = DATA_DIR / folder_name
    h5_path = folder / "retarget_poses_g1.h5"
    out_path = folder / "motion_for_holosoma.npz"

    if not h5_path.exists():
        print("  SKIP: {} not found".format(h5_path))
        return

    print("  Reading {}...".format(h5_path))
    h5 = read_h5(h5_path)
    input_fps = h5["fps"]
    T_in = h5["joints"].shape[0]
    duration = (T_in - 1) / input_fps
    print("    Input: {} frames @ {} fps = {:.2f}s, {} joints".format(
        T_in, input_fps, duration, h5["joints"].shape[1]))

    # 1. Pad 23 joints → 29 joints
    joints_29 = pad_joints_23_to_29(h5["joints"], h5["h5_joint_names"])

    # 2. Convert root_quat from xyzw → wxyz
    root_quat_wxyz = xyzw_to_wxyz(h5["root_quat"])  # (T_in, 4)
    root_pos = h5["root_pos"]  # (T_in, 3)

    # 3. Build qpos at input FPS: [root_xyz, root_wxyz, joints_29]
    qpos_in = np.concatenate([root_pos, root_quat_wxyz, joints_29], axis=1)  # (T_in, 36)
    assert qpos_in.shape == (T_in, 36), "qpos shape mismatch: {}".format(qpos_in.shape)

    # 4. Upsample from input_fps to OUTPUT_FPS via interpolation
    times_in = np.arange(T_in) / input_fps
    T_out = int(np.ceil(duration * OUTPUT_FPS)) + 1
    times_out = np.linspace(0.0, duration, T_out)
    print("    Upsampling: {} fps → {} fps ({} → {} frames)".format(
        int(input_fps), OUTPUT_FPS, T_in, T_out))

    # Interpolate root position and joint angles (linear)
    root_pos_up = interpolate_positions(root_pos, times_in, times_out)       # (T_out, 3)
    joints_29_up = interpolate_positions(joints_29, times_in, times_out)     # (T_out, 29)

    # Interpolate root quaternion (slerp)
    root_quat_up = interpolate_quaternions(root_quat_wxyz, times_in, times_out)  # (T_out, 4) wxyz

    # Assemble upsampled qpos
    qpos_up = np.concatenate([root_pos_up, root_quat_up, joints_29_up], axis=1)  # (T_out, 36)

    # 5. Run MuJoCo FK for all upsampled frames
    dt_out = 1.0 / OUTPUT_FPS
    print("    Running MuJoCo forward kinematics for {} frames...".format(T_out))
    fk = mujoco_forward_kinematics(model, qpos_up, dt_out)

    # 6. Compute joint velocities via numerical differentiation
    joint_vel = compute_velocities_numerical(qpos_up, dt_out)  # (T_out, 35)

    # 7. Get body/joint names from MuJoCo model
    joint_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    body_names = []
    for i in range(model.nbody):
        body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

    # 8. Save
    print("    Saving to {}...".format(out_path))
    np.savez(
        out_path,
        fps=np.array([OUTPUT_FPS]),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        joint_pos=qpos_up,                      # (T_out, 36)
        joint_vel=joint_vel,                     # (T_out, 35)
        body_pos_w=fk["body_pos_w"],             # (T_out, 51, 3)
        body_quat_w=fk["body_quat_w"],           # (T_out, 51, 4) wxyz
        body_lin_vel_w=fk["body_lin_vel_w"],      # (T_out, 51, 3)
        body_ang_vel_w=fk["body_ang_vel_w"],      # (T_out, 51, 3)
    )
    print("    Done: {} → shape summary:".format(out_path.name))
    print("      joint_pos:    {}".format(qpos_up.shape))
    print("      joint_vel:    {}".format(joint_vel.shape))
    print("      body_pos_w:   {}".format(fk["body_pos_w"].shape))
    print("      body_quat_w:  {}".format(fk["body_quat_w"].shape))
    print("      body_lin_vel: {}".format(fk["body_lin_vel_w"].shape))
    print("      body_ang_vel: {}".format(fk["body_ang_vel_w"].shape))


def main():
    parser = argparse.ArgumentParser(description="Convert H5 motion to Holosoma .npz")
    parser.add_argument(
        "folders", nargs="*",
        help="Folder name(s) under data/. If empty, converts all.",
    )
    args = parser.parse_args()

    all_folders = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    folders = args.folders if args.folders else all_folders

    print("Loading MuJoCo model: {}".format(MUJOCO_MODEL_PATH))
    model = mujoco.MjModel.from_xml_path(str(MUJOCO_MODEL_PATH))
    print("  nq={}, nv={}, nbody={}".format(model.nq, model.nv, model.nbody))

    for folder_name in folders:
        print("\n=== {} ===".format(folder_name))
        convert_one(folder_name, model)

    print("\nAll done.")


if __name__ == "__main__":
    main()
