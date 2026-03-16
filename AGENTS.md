# Repository Guidelines

## Project Structure & Module Organization
- `src/holosoma/`: core training/eval framework (PPO/FastSAC, IsaacGym/IsaacSim/MJWarp).
- `src/holosoma_inference/`: deployment and sim-to-sim / sim-to-real policy runtime.
- `src/holosoma_retargeting/`: motion retargeting pipeline.
- `scripts/`: environment setup and workflow entrypoints (`setup_*`, `source_*`, train/eval helpers).
- `tests/`: `ci/` shell entrypoints, `e2e/` Python integration tests, `nightly/` long-run checks.
- `data/`, `tmp/`, `logs/`: local assets and outputs (often large/generated; keep changes intentional).

## Build, Test, and Development Commands
- Setup environments (choose one):
  - `bash scripts/setup_isaacgym.sh`
  - `bash scripts/setup_isaacsim.sh`
  - `bash scripts/setup_mujoco.sh`
- Activate runtime before commands:
  - `source scripts/source_isaacgym_setup.sh` or `source scripts/source_isaacsim_setup.sh`
- Train example:
  - `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:isaacgym logger:wandb`
- Replay motion (WBT):
  - `python src/holosoma/holosoma/replay.py exp:g1-29dof-wbt --training.num-envs=1`
- Test:
  - `pytest -s --ignore=thirdparty -m "not isaacsim"`
  - `python -m pytest -s -m "isaacsim" --ignore=thirdparty`

## Coding Style & Naming Conventions
- Python style: 4-space indent, max line length 120, type-aware dataclass configs.
- Linting/formatting uses Ruff (`pyproject.toml`); run `ruff check .` (and `--fix` when safe).
- Naming:
  - `snake_case` for modules/functions/variables.
  - `PascalCase` for classes.
  - Tyro preset names use prefixes like `exp:`, `simulator:`, `terrain:` with kebab-case values.

## Testing Guidelines
- Add tests under `tests/e2e/` for workflow-level behavior; keep runtime short (small env count / iterations).
- Use `@pytest.mark.isaacsim` for IsaacSim-only tests.
- Test file names should be `test_*.py`; assert observable behavior (checkpoint creation, eval startup, FPS/log signals).

## Commit & Pull Request Guidelines
- Follow current history style: short imperative subject lines (e.g., `Add ...`, `Revise ...`), optional scope tags, optional issue refs (`#27`).
- PRs should include:
  - what changed and why,
  - exact run/test commands used,
  - affected simulator/task/configs,
  - logs/videos/screenshots for behavior changes.
- Avoid unrelated reformatting; keep diffs focused.

## Security & Configuration Tips
- Do not commit secrets (e.g., `WANDB_API_KEY`).
- Prefer environment variables for credentials and simulator-specific runtime flags.
- For security issues, follow `CONTRIBUTING.md` disclosure guidance (do not file public exploit details).

## IsaacSim Critical Pitfall
- For WBT resets, avoid relying on indexed articulation writes (`env_ids` / `joint_ids`) for robot root/DOF state sync in IsaacSim; this can silently keep default states and cause immediate `bad_tracking` termination.
- Use full-state writes for reset sync:
  - Root: update full root-state tensor and write all envs.
  - DOF: map desired DOFs into full articulation-order joint tensors, then write all joints/envs.
- In `MotionCommand.reset`, write root states by full-row assignment (not partial slice assignment) to keep `RootStatesProxy` xyzw/wxyz buffers consistent.
