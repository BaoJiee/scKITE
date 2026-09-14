#!/usr/bin/env bash
set -euo pipefail





SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"








if [[ -n "${SCKITE_CONDA_ENV:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: SCKITE_CONDA_ENV is set but conda is unavailable."
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${SCKITE_CONDA_ENV}"
fi







export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"


NPROC_PER_NODE="${NPROC_PER_NODE:-2}"






export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export PYTHONUNBUFFERED=1


export PYTORCH_ALLOC_CONF=expandable_segments:True







CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config.yaml}"


TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
TRAIN_MODULE="sckite.stage1.train"


LOG_DIR="${PROJECT_DIR}/outputs/stage1/logs"

















RESUME=""








mkdir -p "${LOG_DIR}"






if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: CONFIG_PATH 不存在：${CONFIG_PATH}"
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: TRAIN_SCRIPT 不存在：${TRAIN_SCRIPT}"
    exit 1
fi


if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
    echo "ERROR: RESUME checkpoint 不存在：${RESUME}"
    exit 1
fi





RUN_ARGS=(
    --module "${TRAIN_MODULE}"
    --config "${CONFIG_PATH}"
)


if [[ -n "${RESUME}" ]]; then
    RUN_ARGS+=(
        --resume "${RESUME}"
    )
fi





LOG_FILE="${LOG_DIR}/stage1_encoder_$(date +%Y%m%d_%H%M%S).log"


echo "============================================================" | tee -a "${LOG_FILE}"
echo "scKITE Stage 1 Encoder Pretraining" | tee -a "${LOG_FILE}"
echo "Masked-Expression Reconstruction" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

echo "Training started at: $(date)" | tee -a "${LOG_FILE}"

echo "PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}"
echo "CONFIG_PATH=${CONFIG_PATH}" | tee -a "${LOG_FILE}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}" | tee -a "${LOG_FILE}"

echo "RESUME=${RESUME:-<none>}" | tee -a "${LOG_FILE}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}" | tee -a "${LOG_FILE}"

echo "LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "============================================================" | tee -a "${LOG_FILE}"






if torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${RUN_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
then

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "Training finished successfully at $(date)" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

else

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "Training failed at $(date)" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    exit 1

fi
