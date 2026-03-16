"""Standalone kinematic motion replay script using MuJoCo + WandB.

Loads an npz motion file and a scene OBJ, plays back the motion frame-by-frame
(no physics, pure kinematic), saves a video, and uploads to WandB.

Usage:
    WANDB_API_KEY=... MUJOCO_GL=egl python scripts/visualize_motion_mujoco.py \
        --motion data/stairs/motion_for_holosoma.npz \
        --scene  data/stairs/scene_mesh_gravity_aligned.obj \
        --output /tmp/replay_stairs.mp4
"""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import wandb

# ──────────────────────────────────────────────────────────────
# Paths (defaults can be overridden via CLI)
# ──────────────────────────────────────────────────────────────
ROOT = "/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma"
ROBOT_XML = os.path.join(
    ROOT, "src/holosoma/holosoma/data/robots/g1/g1_29dof.xml"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True, help="Path to motion_for_holosoma.npz")
    p.add_argument("--scene", required=True, help="Path to scene_mesh_gravity_aligned.obj")
    p.add_argument("--output", default="/tmp/replay.mp4", help="Output video path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--wandb-project", default="holosoma-replay")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true", help="Skip WandB upload")
    return p.parse_args()


def build_model(scene_obj_path: str, width: int = 1280, height: int = 720) -> mujoco.MjModel:
    """Load G1 XML, inject the scene OBJ mesh, return compiled model."""
    # Resolve to absolute path so MuJoCo doesn't interpret it relative to meshdir
    scene_obj_path = os.path.abspath(scene_obj_path)

    # Parse and modify XML in-memory
    ET.register_namespace("", "")
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()

    # ── asset: add the scene mesh ──────────────────────────────
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", attrib={"name": "scene_mesh", "file": scene_obj_path})

    # ── worldbody: add scene geom (visual + collision) ─────────
    worldbody = root.find("worldbody")
    if worldbody is not None:
        ET.SubElement(
            worldbody,
            "geom",
            attrib={
                "name": "scene_geom",
                "type": "mesh",
                "mesh": "scene_mesh",
                "contype": "1",
                "conaffinity": "1",
                "rgba": "0.65 0.65 0.65 1",
            },
        )

    # Set offscreen framebuffer to match requested render resolution
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_elem = visual.find("global")
    if global_elem is None:
        global_elem = ET.SubElement(visual, "global")
    global_elem.set("offwidth", str(width))
    global_elem.set("offheight", str(height))

    # Remove the <contact> section — it references a "floor" geom that only exists
    # in the training world (not in our custom scene model).
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    # Write to a temp file next to the robot XML so meshdir="./meshes/" still resolves
    robot_dir = os.path.dirname(ROBOT_XML)
    tmp_path = os.path.join(robot_dir, "_tmp_g1_scene.xml")
    tree.write(tmp_path)

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.remove(tmp_path)

    return model


def make_camera(model: mujoco.MjModel, mj_data: mujoco.MjData) -> mujoco.MjvCamera:
    """Create a camera that looks at the scene from a fixed angle."""
    # Find the pelvis body to centre the view
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_pelvis")
    if pelvis_id < 0:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 5.0
    cam.azimuth = -135.0   # front-right view
    cam.elevation = -20.0  # slightly above
    # Track the robot pelvis
    if pelvis_id >= 0:
        cam.lookat[:] = mj_data.xpos[pelvis_id]
    else:
        cam.lookat[:] = mj_data.qpos[:3]
    return cam


def render_motion(args: argparse.Namespace) -> str:
    """Main rendering loop. Returns path to saved video."""
    # ── Load motion data ───────────────────────────────────────
    print(f"Loading motion data from: {args.motion}")
    with np.load(args.motion) as d:
        qpos_seq = d["joint_pos"]   # (T, 36): [root_pos(3), root_quat_wxyz(4), joints(29)]
        fps = float(d["fps"][0])
        body_names = list(d["body_names"])

    T = len(qpos_seq)
    print(f"  {T} frames @ {fps:.0f} fps  ({T / fps:.1f} s)")

    # ── Build MuJoCo model ─────────────────────────────────────
    print(f"Building MuJoCo model with scene: {args.scene}")
    model = build_model(args.scene, width=args.width, height=args.height)
    mj_data = mujoco.MjData(model)
    print(f"  Model: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

    # Warm-up forward pass to populate xpos
    mj_data.qpos[:] = qpos_seq[0]
    mujoco.mj_forward(model, mj_data)

    # ── Renderer ───────────────────────────────────────────────
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera(model, mj_data)

    # ── Video writer ───────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (args.width, args.height))

    print(f"Rendering frames → {args.output} ...")
    for t in range(T):
        mj_data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(model, mj_data)

        # Smoothly update camera lookat to follow the robot's pelvis
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_pelvis")
        if pelvis_id < 0:
            pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id >= 0:
            alpha = 0.05  # smooth tracking
            cam.lookat[:] = (1 - alpha) * cam.lookat + alpha * mj_data.xpos[pelvis_id]

        renderer.update_scene(mj_data, camera=cam)
        frame_rgb = renderer.render()
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if t % 50 == 0:
            print(f"  [{t:4d}/{T}]  pelvis_z = {mj_data.qpos[2]:.3f}")

    writer.release()
    print(f"Video saved: {args.output}")
    return args.output


def upload_wandb(video_path: str, args: argparse.Namespace) -> None:
    scene_name = os.path.basename(os.path.dirname(args.motion))
    run_name = args.wandb_name or f"replay-{scene_name}"

    print(f"Uploading to WandB project={args.wandb_project!r} run={run_name!r} ...")
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={"motion": args.motion, "scene": args.scene},
    )
    run.log({"replay_video": wandb.Video(video_path, fps=50, format="mp4")})
    run.finish()
    print(f"WandB run URL: {run.url}")


def main() -> None:
    args = parse_args()
    video_path = render_motion(args)

    if not args.no_wandb:
        upload_wandb(video_path, args)


if __name__ == "__main__":
    main()
