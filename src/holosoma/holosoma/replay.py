from __future__ import annotations

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


def replay(tyro_config: ExperimentConfig):
    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    recorder = env.simulator.video_recorder
    if recorder is not None and recorder.enabled and not recorder.is_recording:
        recorder.start_recording(episode_id=0)

    done = False
    while not done:
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
        if recorder is not None and recorder.enabled:
            env.simulator.capture_video_frame(env_id=0)

    if recorder is not None and recorder.enabled and recorder.is_recording:
        recorder.stop_recording()

    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
