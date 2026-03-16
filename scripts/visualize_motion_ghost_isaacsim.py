#!/usr/bin/env python3
"""Kinematic "ghost" motion visualization in Isaac Sim.

This script visualizes motion from NPZ files that contain both:
- motion arrays (joint/root/body trajectories)
- embedded scene mesh (`mesh_vertices`, `mesh_faces`)

Unlike `holosoma.replay`, this script does not advance physics for replay.
It writes each frame's pose directly, renders, and records video.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_values.logger import disabled
from holosoma.config_values.robot import g1_29dof_diverse_action_scale
from holosoma.config_values.terrain import terrain_load_obj
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_future_motion
from holosoma.utils.common import seeding
from holosoma.utils.eval_utils import init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app


def _disable_global_gravity(simulator) -> None:
    """Disable gravity in the active PhysicsScene."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        scene_api = UsdPhysics.Scene(prim)
        scene_api.CreateGravityMagnitudeAttr(0.0)
        scene_api.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
        physx_scene_api = PhysxSchema.PhysxSceneAPI(prim)
        # Keep playback stable for kinematic visualization (no residual drift from contacts).
        physx_scene_api.CreateMinPositionIterationCountAttr(1)
        physx_scene_api.CreateMaxPositionIterationCountAttr(1)
        physx_scene_api.CreateMinVelocityIterationCountAttr(0)
        physx_scene_api.CreateMaxVelocityIterationCountAttr(0)
        break


def _build_replay_config(
    motion_file: str,
    save_dir: str,
    env_origin_in_tile: list[float],
    playback_rate: float,
    robot_asset_root: str | None,
) -> ExperimentConfig:
    """Build a minimal, deterministic IsaacSim config for ghost replay."""
    base = g1_29dof_wbt_future_motion

    # Update motion command file and disable training-time transitions/noise.
    motion_term = base.command.setup_terms["motion_command"]
    motion_cfg = motion_term.params["motion_config"]
    motion_cfg = replace(
        motion_cfg,
        motion_file=motion_file,
        enable_default_pose_prepend=False,
        enable_default_pose_append=False,
        start_at_timestep_zero_prob=1.0,
        freeze_at_timestep_zero_prob=0.0,
        noise_to_initial_pose=replace(motion_cfg.noise_to_initial_pose, overall_noise_scale=0.0),
    )
    command_cfg = replace(
        base.command,
        setup_terms={
            **base.command.setup_terms,
            "motion_command": replace(motion_term, params={"motion_config": motion_cfg}),
        },
    )

    # Use mesh embedded in NPZ for terrain.
    terrain_cfg = replace(
        terrain_load_obj,
        terrain_term=replace(
            terrain_load_obj.terrain_term,
            obj_file_path=motion_file,
            tile_mesh=False,
            env_origin_in_tile=env_origin_in_tile,
        ),
    )

    # Enable local video recording only.
    camera_cfg = replace(
        disabled.video.camera,
        offset=[1.5, -2.0, 1.0],
        target_offset=[0.0, 0.0, 0.9],
        tracking_body_name="torso_link",
    )
    video_cfg = replace(
        disabled.video,
        enabled=True,
        interval=1,
        upload_to_wandb=False,
        show_command_overlay=False,
        save_dir=save_dir,
        playback_rate=playback_rate,
        camera=camera_cfg,
    )
    logger_cfg = replace(disabled, video=video_cfg)

    # Empty randomization manager to keep kinematic replay deterministic.
    randomization_cfg = RandomizationManagerCfg()

    sim_cfg = replace(
        base.simulator,
        config=replace(
            base.simulator.config,
            scene=replace(base.simulator.config.scene, env_spacing=0.0),
            sim=replace(
                base.simulator.config.sim,
                control_decimation=1,
                render_interval=1,
            ),
        ),
    )

    robot_cfg = g1_29dof_diverse_action_scale
    if robot_asset_root:
        robot_cfg = replace(
            robot_cfg,
            asset=replace(robot_cfg.asset, asset_root=robot_asset_root),
        )

    return replace(
        base,
        simulator=sim_cfg,
        terrain=terrain_cfg,
        command=command_cfg,
        robot=robot_cfg,
        logger=logger_cfg,
        randomization=randomization_cfg,
        training=replace(base.training, num_envs=1, headless=True),
    )


def _apply_motion_frame(env, motion_command, frame_idx: int) -> None:
    """Write one frame of motion into simulator state tensors."""
    motion_command.time_steps[:] = frame_idx

    env_ids = torch.arange(env.num_envs, device=env.device)

    root_pos = motion_command.root_pos_w.clone()
    root_ori = motion_command.root_quat_w.clone()  # xyzw
    root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()
    joint_pos = motion_command.joint_pos.clone()
    joint_vel = motion_command.joint_vel.clone()

    env.simulator.dof_pos[env_ids].copy_(joint_pos)
    env.simulator.dof_vel[env_ids].copy_(joint_vel)

    target_root_states = env.simulator.robot_root_states[env_ids].clone()
    target_root_states[:, :3] = root_pos
    target_root_states[:, 3:7] = root_ori
    target_root_states[:, 7:10] = root_lin_vel
    target_root_states[:, 10:13] = root_ang_vel
    env.simulator.robot_root_states[env_ids] = target_root_states

    env.simulator.set_actor_root_state_tensor(env_ids, env.simulator.all_root_states)
    env.simulator.set_dof_state_tensor(env_ids, env.simulator.dof_state)

    if motion_command.motion.has_object:
        object_states = torch.zeros(len(env_ids), 13, device=env.device)
        object_states[:, :3] = motion_command.object_pos_w.clone()
        object_states[:, 3:7] = motion_command.object_quat_w.clone()
        env.simulator.set_actor_states(["object"], env_ids, object_states)


def replay_ghost(
    motion_file: Path,
    save_dir: Path,
    env_origin_in_tile: list[float],
    playback_rate: float,
    robot_asset_root: str | None,
) -> None:
    config = _build_replay_config(
        motion_file=str(motion_file),
        save_dir=str(save_dir),
        env_origin_in_tile=env_origin_in_tile,
        playback_rate=playback_rate,
        robot_asset_root=robot_asset_root,
    )

    simulation_app = init_sim_imports(config)
    try:
        seeding(42, torch_deterministic=False)
        env_cls = get_class(config.env_class)
        env = env_cls(get_tyro_env_config(config), device=("cuda:0" if torch.cuda.is_available() else "cpu"))
        _disable_global_gravity(env.simulator)

        motion_command = env.command_manager.get_state("motion_command")
        total_frames = int(motion_command.motion.time_step_total)
        sim_dt = 1.0 / float(env.simulator.simulator_config.sim.fps)

        recorder = env.simulator.video_recorder
        if recorder is None or not recorder.enabled:
            raise RuntimeError("Video recorder is not enabled.")
        if not recorder.is_recording:
            recorder.start_recording(episode_id=0)

        for t in range(total_frames):
            _apply_motion_frame(env, motion_command, t)

            # Render authored pose first (before any physics integration).
            env.simulator.scene.write_data_to_sim()
            env.simulator.render(sync_frame_time=False)
            env.simulator.capture_video_frame(env_id=0)

            # Advance app/replicator one tick for the next frame.
            env.simulator.sim.step(render=False)
            env.simulator.scene.update(dt=sim_dt)
            env.simulator.refresh_sim_tensors()

        if recorder.is_recording:
            recorder.stop_recording()
    finally:
        close_simulation_app(simulation_app)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render kinematic ghost replay videos in Isaac Sim.")
    parser.add_argument(
        "--motion",
        action="append",
        required=True,
        help="Path to motion+mesh NPZ. Repeat flag for multiple files.",
    )
    parser.add_argument(
        "--output-root",
        default="tmp/replay_scene_motion_ghost_isaacsim",
        help="Directory where per-motion replay videos are saved.",
    )
    parser.add_argument(
        "--env-origin-in-tile",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Terrain env-origin-in-tile xyz offset.",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=0.25,
        help="Video playback rate multiplier.",
    )
    parser.add_argument(
        "--robot-asset-root",
        default="/tmp/holosoma_robot_assets",
        help="Asset root override for robot conversion cache/write permission issues.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for motion in args.motion:
        motion_path = Path(motion).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        run_dir = output_root / motion_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ghost replay start: {motion_path}")
        replay_ghost(
            motion_file=motion_path,
            save_dir=run_dir,
            env_origin_in_tile=args.env_origin_in_tile,
            playback_rate=args.playback_rate,
            robot_asset_root=args.robot_asset_root,
        )
        print(f"[INFO] Ghost replay done: {motion_path} -> {run_dir}")


if __name__ == "__main__":
    main()
