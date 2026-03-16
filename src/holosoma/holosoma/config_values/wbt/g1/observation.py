"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

# Future motion observation config
# Includes future_motion_targets for motion encoder
future_motion_obs_group = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "future_motion_targets": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:future_motion_targets",
            scale=1.0,
            noise=0.0,
            params={"future_num_steps": 20, "future_max_steps": 95},
        ),
    },
)

g1_29dof_wbt_observation_future_motion = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
    },
)

# Future motion without local key body positions (ablation)
future_motion_obs_group_no_key_body = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "future_motion_targets": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:future_motion_targets",
            scale=1.0,
            noise=0.0,
            params={
                "future_num_steps": 20,
                "future_max_steps": 95,
                "include_key_body_pos": False,
            },
        ),
    },
)

g1_29dof_wbt_observation_future_motion_no_key_body = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group_no_key_body,
    },
)

actor_state_history_videomimic_stage1 = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=5,
    terms={
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "projected_gravity": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:projected_gravity",
            scale=1.0,
            noise=0.05,
        ),
    },
)

actor_tracking_history_videomimic_stage1 = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=5,
    terms={
        "torso_xy_rel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:torso_xy_rel",
            scale=1.0,
            noise=0.02,
        ),
        "torso_yaw_rel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:torso_yaw_rel",
            scale=1.0,
            noise=0.05,
        ),
    },
)

actor_targets_videomimic_stage1 = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "target_joint_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:target_joint_pos",
            scale=1.0,
            noise=0.0,
        ),
        "target_root_roll_pitch": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:target_root_roll_pitch",
            scale=1.0,
            noise=0.0,
        ),
    },
)

critic_obs_videomimic_stage1 = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.0,
        ),
        "base_lin_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_lin_vel",
            scale=1.0,
            noise=0.0,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.0,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.0,
        ),
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "projected_gravity": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:projected_gravity",
            scale=1.0,
            noise=0.0,
        ),
        "robot_body_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "robot_body_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "target_joint_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:target_joint_pos",
            scale=1.0,
            noise=0.0,
        ),
        "target_root_roll_pitch": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:target_root_roll_pitch",
            scale=1.0,
            noise=0.0,
        ),
        "torso_xy_rel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:torso_xy_rel",
            scale=1.0,
            noise=0.0,
        ),
        "torso_yaw_rel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:torso_yaw_rel",
            scale=1.0,
            noise=0.0,
        ),
    },
)

g1_29dof_wbt_observation_lafan_videomimic_stage1 = ObservationManagerCfg(
    groups={
        "actor_state_history": actor_state_history_videomimic_stage1,
        "actor_tracking_history": actor_tracking_history_videomimic_stage1,
        "actor_targets": actor_targets_videomimic_stage1,
        "critic_obs": critic_obs_videomimic_stage1,
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_future_motion",
    "g1_29dof_wbt_observation_future_motion_no_key_body",
    "g1_29dof_wbt_observation_lafan_videomimic_stage1",
]
