#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/Gene_Perturbation_Prediction"
TRAIN_SCRIPT="${PROJECT_ROOT}/train.py"
PYTHON_BIN="/root/miniconda3/envs/cllm/bin/python"

DATA_ROOT="${PROJECT_ROOT}/data"
DATASET_NAME="replogle_exp7"
DATASET_DIR="${DATA_ROOT}/${DATASET_NAME}"
INPUT_H5AD="${DATASET_DIR}/perturb_processed.h5ad"

SAVE_ROOT="${PROJECT_ROOT}/results"
STAGE2_CKPT_PATH="/root/autodl-tmp/pt/geosketch_50_encoderonly.pt"
STAGE2_VOCAB_PATH="/root/autodl-tmp/vocab/gene_vocabulary.jsonl"
STAGE2_MODEL_PY_PATH="/root/autodl-tmp/ExpertCoder/ExpertCoder_Separate_decoder_activate_regulon/model.py"
STAGE2_MODEL_KWARGS_JSON=""

# Stage2/scFM configuration. adata.X is already log1p, so as_is avoids log1p twice.
SCFM_MODE="contextual"
PERT_MODE="native"
FREEZE_PERT_EMB="false"
PROJECT_METHOD="mean_pool"
MISSING_STRATEGY="mean_gene"
NORMALIZE_EMBEDDING="false"
CONTEXTUAL_VALUE_MODE="as_is"
NUM_BINS="51"
CONTEXTUAL_MAX_GENES="0"
CONTEXTUAL_GENE_SELECTION="matched_first"
CONTEXTUAL_FALLBACK="static"

CUDA_DEVICE="0"
SEED="1"

# replogle_exp7 contains only single-gene perturbations. Reuse the verified split.
SPLIT="single"
TRAIN_GENE_SET_SIZE="0.75"
COMBO_SEEN2_TRAIN_FRAC="0.75"

EPOCHS="15"
# Start conservatively in contextual mode; increase only after checking GPU memory.
BATCH_SIZE="400"
TEST_BATCH_SIZE="400"
LR="1e-5"
WEIGHT_DECAY="5e-4"

HIDDEN_SIZE="512"
NUM_GO_GNN_LAYERS="1"
NUM_GENE_GNN_LAYERS="1"
DECODER_HIDDEN_SIZE="16"
NUM_SIMILAR_GENES_GO_GRAPH="20"
NUM_SIMILAR_GENES_COEXPRESS_GRAPH="20"
COEXPRESS_THRESHOLD="0.4"
UNCERTAINTY="false"
UNCERTAINTY_REG="1"
DIRECTION_LAMBDA="1e-1"

WANDB="true"
WANDB_PROJECT="run_scfm_gears_replogle_exp7"
WANDB_MODE="online"

# Keep this false for the first run. Change to true only if automatic shutdown is desired.
SHUTDOWN_AFTER_FINISH="true"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export WANDB_MODE="${WANDB_MODE}"

cd "${PROJECT_ROOT}"

for required_path in \
  "${TRAIN_SCRIPT}" \
  "${PYTHON_BIN}" \
  "${INPUT_H5AD}" \
  "${DATASET_DIR}/data_pyg/cell_graphs.pkl" \
  "${STAGE2_CKPT_PATH}" \
  "${STAGE2_VOCAB_PATH}" \
  "${STAGE2_MODEL_PY_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

if [[ "${SCFM_MODE}" == "static" ]]; then
  GENE_EMBEDDING_MODE="scfm_static"
elif [[ "${SCFM_MODE}" == "contextual" ]]; then
  GENE_EMBEDDING_MODE="scfm_contextual"
else
  echo "Invalid SCFM_MODE=${SCFM_MODE}; use static or contextual." >&2
  exit 1
fi

if [[ "${SCFM_MODE}" == "contextual" ]]; then
  echo "[Warning] Contextual mode processes ${CONTEXTUAL_MAX_GENES:-all} selected genes per cell."
  echo "[Warning] Reduce BATCH_SIZE/TEST_BATCH_SIZE if CUDA runs out of memory."
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="geosketch_50_encoderonly_${DATASET_NAME}_${SCFM_MODE}_${CONTEXTUAL_VALUE_MODE}_max${CONTEXTUAL_MAX_GENES}_pert${PERT_MODE}_hs${HIDDEN_SIZE}_seed${SEED}_${TIMESTAMP}"
SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"
LOG_DIR="${SAVE_DIR}/logs"
LOG_FILE="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"

echo "Run name: ${RUN_NAME}"
echo "Dataset: ${DATASET_DIR}"
echo "Split: ${SPLIT}"
echo "SCFM mode: ${SCFM_MODE}"
echo "Gene embedding mode: ${GENE_EMBEDDING_MODE}"
echo "Perturbation mode: ${PERT_MODE}"
echo "Batch size: ${BATCH_SIZE}; test batch size: ${TEST_BATCH_SIZE}"
echo "Save directory: ${SAVE_DIR}"
echo "Log file: ${LOG_FILE}"

ARGS=(
  --data_root "${DATA_ROOT}"
  --dataset_dir "${DATASET_DIR}"
  --save_dir "${SAVE_DIR}"
  --device cuda
  --seed "${SEED}"
  --split "${SPLIT}"
  --train_gene_set_size "${TRAIN_GENE_SET_SIZE}"
  --combo_seen2_train_frac "${COMBO_SEEN2_TRAIN_FRAC}"
  --batch_size "${BATCH_SIZE}"
  --test_batch_size "${TEST_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --hidden_size "${HIDDEN_SIZE}"
  --num_go_gnn_layers "${NUM_GO_GNN_LAYERS}"
  --num_gene_gnn_layers "${NUM_GENE_GNN_LAYERS}"
  --decoder_hidden_size "${DECODER_HIDDEN_SIZE}"
  --num_similar_genes_go_graph "${NUM_SIMILAR_GENES_GO_GRAPH}"
  --num_similar_genes_co_express_graph "${NUM_SIMILAR_GENES_COEXPRESS_GRAPH}"
  --coexpress_threshold "${COEXPRESS_THRESHOLD}"
  --uncertainty "${UNCERTAINTY}"
  --uncertainty_reg "${UNCERTAINTY_REG}"
  --direction_lambda "${DIRECTION_LAMBDA}"
  --no_perturb false
  --adapter_name stage2
  --stage2_ckpt_path "${STAGE2_CKPT_PATH}"
  --stage2_vocab_path "${STAGE2_VOCAB_PATH}"
  --stage2_model_py_path "${STAGE2_MODEL_PY_PATH}"
  --num_bins "${NUM_BINS}"
  --project_method "${PROJECT_METHOD}"
  --missing_strategy "${MISSING_STRATEGY}"
  --normalize_embedding "${NORMALIZE_EMBEDDING}"
  --gene_embedding_mode "${GENE_EMBEDDING_MODE}"
  --pert_embedding_mode "${PERT_MODE}"
  --freeze_pert_emb "${FREEZE_PERT_EMB}"
  --wandb "${WANDB}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${RUN_NAME}"
  --contextual_value_mode "${CONTEXTUAL_VALUE_MODE}"
  --contextual_max_genes "${CONTEXTUAL_MAX_GENES}"
  --contextual_gene_selection "${CONTEXTUAL_GENE_SELECTION}"
  --contextual_fallback "${CONTEXTUAL_FALLBACK}"
)

if [[ -n "${STAGE2_MODEL_KWARGS_JSON}" ]]; then
  ARGS+=(--stage2_model_kwargs_json "${STAGE2_MODEL_KWARGS_JSON}")
fi

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

if [[ "${SHUTDOWN_AFTER_FINISH}" == "true" ]]; then
  shutdown -h now
fi
