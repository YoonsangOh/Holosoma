WORKSPACE_DIR=$HOME/.holosoma_deps
CONDA_ROOT=$WORKSPACE_DIR/miniconda3

# Prefer this repository's source tree over any globally editable install.
# This avoids importing a different clone (e.g. /work/holosoma) by accident.
if [ -n "${SCRIPT_DIR:-}" ]; then
    REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
    LOCAL_SRC_PATHS=(
        "${REPO_ROOT}/src/holosoma"
        "${REPO_ROOT}/src/holosoma_inference"
        "${REPO_ROOT}/src/holosoma_retargeting"
    )
    for SRC_PATH in "${LOCAL_SRC_PATHS[@]}"; do
        if [ -d "${SRC_PATH}" ]; then
            case ":${PYTHONPATH:-}:" in
                *":${SRC_PATH}:"*) ;;
                *) export PYTHONPATH="${SRC_PATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
            esac
        fi
    done
fi
