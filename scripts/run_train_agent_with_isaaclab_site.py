#!/usr/bin/env python3
"""Launch Holosoma training with the local source tree plus an external IsaacLab site-packages.

This is a thin compatibility wrapper for machines where:
- the Holosoma training environment has the right Python dependencies (pydantic/tyro/etc.), but
- IsaacSim / IsaacLab live in a different conda environment.

The wrapper keeps Holosoma's Python packages first on ``sys.path`` so their dependency versions
win, then adds the IsaacLab site-packages directory so IsaacSim can be imported at runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import site
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SRC_PATHS = [
    REPO_ROOT / "src/holosoma",
    REPO_ROOT / "src/holosoma_inference",
    REPO_ROOT / "src/holosoma_retargeting",
]
DEFAULT_ISAACLAB_SITE = "/home/nas4_user/kyungminlee/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages"


def _prepend_local_sources() -> None:
    for src_path in reversed(LOCAL_SRC_PATHS):
        src_str = str(src_path)
        if src_path.exists() and src_str not in sys.path:
            sys.path.insert(0, src_str)


def _prepend_library_path(lib_path: Path) -> None:
    if not lib_path.exists():
        return

    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current_ld_library_path.split(":") if part]
    lib_path_str = str(lib_path)
    if lib_path_str not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([lib_path_str, *parts]) if parts else lib_path_str


def _normalize_optional_scene_lists(config):
    """Restore runtime semantics for optional scene fields after Tyro parsing."""
    scene_cfg = config.simulator.config.scene
    normalized_scene_cfg = scene_cfg

    if scene_cfg.scene_files == []:
        normalized_scene_cfg = dataclasses.replace(normalized_scene_cfg, scene_files=None)
    if scene_cfg.rigid_objects == []:
        normalized_scene_cfg = dataclasses.replace(normalized_scene_cfg, rigid_objects=None)

    if normalized_scene_cfg is scene_cfg:
        return config

    normalized_sim_cfg = dataclasses.replace(config.simulator.config, scene=normalized_scene_cfg)
    normalized_simulator = dataclasses.replace(config.simulator, config=normalized_sim_cfg)
    return dataclasses.replace(config, simulator=normalized_simulator)


def main() -> None:
    _prepend_local_sources()

    # Import a Holosoma module before IsaacLab paths are added so the training environment's
    # pydantic/tyro stack stays authoritative.
    import holosoma.config_types.env  # noqa: F401

    isaaclab_site = os.environ.get("HOLOSOMA_ISAACLAB_SITE_PACKAGES", DEFAULT_ISAACLAB_SITE)
    isaaclab_site_path = Path(isaaclab_site)
    isaac_env_root = isaaclab_site_path.parents[2]
    _prepend_library_path(isaac_env_root / "lib")
    _prepend_library_path(isaac_env_root / "lib64")
    site.addsitedir(isaaclab_site)

    from isaaclab.app import AppLauncher
    from holosoma.config_values.experiment import AnnotatedExperimentConfig
    from holosoma.train_agent import train
    from holosoma.utils.tyro_utils import TYRO_CONIFG
    import tyro

    original_argv = sys.argv[:]

    # AppLauncher args are meant for IsaacSim startup, not Tyro. Strip them before Tyro parses
    # the training config, then restore them so init_sim_imports() can consume them later.
    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args, tyro_unknown = app_parser.parse_known_args(sys.argv[1:])
    app_launcher_argv = [arg for arg in original_argv[1:] if arg not in tyro_unknown]

    sys.argv = [original_argv[0]] + tyro_unknown
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    tyro_cfg = _normalize_optional_scene_lists(tyro_cfg)

    sys.argv = [original_argv[0]] + app_launcher_argv + tyro_unknown
    train(tyro_cfg)


if __name__ == "__main__":
    main()
