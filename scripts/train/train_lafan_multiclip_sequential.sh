#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-data/unitree_lafan_29dof/lafan_walk_and_dance}"
PROJECT="${PROJECT:-LAFANMotionTracking}"
GROUP="${GROUP:-lafan_multiclip_seq_20260312}"
RUN_PREFIX="${RUN_PREFIX:-lafan_multiclip_seq}"
PYTHON_BIN="${PYTHON_BIN:-/home/nas4_user/kyungminlee/anaconda3/envs/holosoma_yoonsang/bin/python}"
SIMULATOR_PRESET="${SIMULATOR_PRESET:-mujoco}"
NUM_ENVS="${NUM_ENVS:-1}"
ITERATIONS_PER_CLIP="${ITERATIONS_PER_CLIP:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-1}"
IGNORE_UNSUPPORTED_RANDOMIZATION="${IGNORE_UNSUPPORTED_RANDOMIZATION:-True}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
CLIP_LIMIT="${CLIP_LIMIT:-0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

mapfile -t MOTION_FILES < <(find "${DATASET_DIR}" -maxdepth 1 -name '*.npz' | sort)

if [[ "${#MOTION_FILES[@]}" -eq 0 ]]; then
    echo "No motion files found under ${DATASET_DIR}" >&2
    exit 1
fi

if [[ "${CLIP_LIMIT}" -gt 0 && "${CLIP_LIMIT}" -lt "${#MOTION_FILES[@]}" ]]; then
    MOTION_FILES=("${MOTION_FILES[@]:0:${CLIP_LIMIT}}")
fi

TIMESTAMP_UTC="$(date -u +%Y%m%d_%H%M%S)"
CHAIN_DIR="logs/${PROJECT}/multiclip_${TIMESTAMP_UTC}"
mkdir -p "${CHAIN_DIR}"
CHAIN_LOG="${CHAIN_DIR}/checkpoint_chain.txt"
CHECKPOINT_PATH="${START_CHECKPOINT}"

echo "dataset_dir=${DATASET_DIR}" | tee -a "${CHAIN_LOG}"
echo "num_clips=${#MOTION_FILES[@]}" | tee -a "${CHAIN_LOG}"
echo "simulator=${SIMULATOR_PRESET}" | tee -a "${CHAIN_LOG}"
echo "iterations_per_clip=${ITERATIONS_PER_CLIP}" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"

for (( i=0; i<${#MOTION_FILES[@]}; i++ )); do
    MOTION_FILE="${MOTION_FILES[$i]}"
    BASENAME="$(basename "${MOTION_FILE}" .npz)"
    RUN_NAME="${RUN_PREFIX}_$(printf '%02d' "$i")_${BASENAME}"

    CMD=(
        "${PYTHON_BIN}" src/holosoma/holosoma/train_agent.py
        exp:g1-29dof-wbt-future-motion
        "simulator:${SIMULATOR_PRESET}"
        logger:wandb
        --logger.project "${PROJECT}"
        --logger.group "${GROUP}"
        --logger.name "${RUN_NAME}"
        --logger.video.enabled False
        --training.num-envs "${NUM_ENVS}"
        --algo.config.num-learning-iterations "${ITERATIONS_PER_CLIP}"
        --algo.config.save-interval "${SAVE_INTERVAL}"
        --algo.config.num-steps-per-env "${NUM_STEPS_PER_ENV}"
        --algo.config.num-learning-epochs "${NUM_EPOCHS}"
        --algo.config.num-mini-batches "${NUM_MINI_BATCHES}"
        --randomization.ignore-unsupported "${IGNORE_UNSUPPORTED_RANDOMIZATION}"
        --command.setup_terms.motion_command.params.motion_config.motion_file="${MOTION_FILE}"
        --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False
        --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False
    )

    if [[ -n "${CHECKPOINT_PATH}" ]]; then
        CMD+=(--training.checkpoint "${CHECKPOINT_PATH}")
    fi

    if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        EXTRA_ARGS=( ${TRAIN_EXTRA_ARGS} )
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] clip_index=${i} motion_file=${MOTION_FILE}" | tee -a "${CHAIN_LOG}"
    "${CMD[@]}"

    RUN_DIR="$(ls -td "logs/${PROJECT}"/*_"${RUN_NAME}" | head -n 1)"
    CHECKPOINT_PATH="$(find "${RUN_DIR}" -maxdepth 1 -name 'model_*.pt' | sort | tail -n 1)"
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
        echo "No checkpoint produced for ${RUN_NAME}" >&2
        exit 1
    fi

    echo "run_dir=${RUN_DIR}" | tee -a "${CHAIN_LOG}"
    echo "checkpoint=${CHECKPOINT_PATH}" | tee -a "${CHAIN_LOG}"
    echo "" | tee -a "${CHAIN_LOG}"
done

echo "final_checkpoint=${CHECKPOINT_PATH}" | tee -a "${CHAIN_LOG}"
