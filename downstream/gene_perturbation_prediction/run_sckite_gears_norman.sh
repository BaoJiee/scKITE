#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_sckite.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/gene_perturbation_prediction}"
DATASET_NAME="norman_custom"
DATASET_DIR="${DATA_ROOT}/${DATASET_NAME}"
INPUT_H5AD="${INPUT_H5AD:-${DATASET_DIR}/perturb_processed.h5ad}"
FORCE_REBUILD_DATA="false"

SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/gene_perturbation_prediction}"
SCKITE_CHECKPOINT_PATH="${SCKITE_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/stage2/best.pt}"
SCKITE_VOCAB_PATH="${SCKITE_VOCAB_PATH:-${REPO_ROOT}/vocab/global_vocab/gene_table.jsonl}"
SCKITE_MODEL_PATH="${SCKITE_MODEL_PATH:-${REPO_ROOT}/sckite/stage2/model.py}"
SCKITE_MODEL_KWARGS_JSON="${SCKITE_MODEL_KWARGS_JSON:-}"

SCKITE_MODE="contextual"
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
SPLIT="simulation"
TRAIN_GENE_SET_SIZE="0.75"
COMBO_SEEN2_TRAIN_FRAC="0.75"

EPOCHS="15"
BATCH_SIZE="420"
TEST_BATCH_SIZE="400"
LR="1e-4"
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
WANDB_PROJECT="sckite_gears"
WANDB_MODE="online"


export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export WANDB_MODE="${WANDB_MODE}"

cd "${PROJECT_ROOT}"

if [[ "${SCKITE_MODE}" == "static" ]]; then
  GENE_EMBEDDING_MODE="sckite_static"
elif [[ "${SCKITE_MODE}" == "contextual" ]]; then
  GENE_EMBEDDING_MODE="sckite_contextual"
else
  echo "Invalid SCKITE_MODE=${SCKITE_MODE}, please use static or contextual."
  exit 1
fi

if [[ "${SCKITE_MODE}" == "contextual" ]]; then
  echo "[Warning] contextual mode is memory-heavy; BATCH_SIZE=1 is recommended for first test."
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="sckite_stage2_${DATASET_NAME}_${SCKITE_MODE}_${CONTEXTUAL_VALUE_MODE}_${CONTEXTUAL_MAX_GENES}_pert${PERT_MODE}_hs${HIDDEN_SIZE}_seed${SEED}_${TIMESTAMP}"
SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"
LOG_DIR="${SAVE_DIR}/logs"
LOG_FILE="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"

echo "Run name: ${RUN_NAME}"
echo "scKITE mode: ${SCKITE_MODE}"
echo "Gene embedding mode: ${GENE_EMBEDDING_MODE}"
echo "Perturbation mode: ${PERT_MODE}"
echo "Save dir: ${SAVE_DIR}"
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
  --adapter_name sckite
  --sckite_checkpoint_path "${SCKITE_CHECKPOINT_PATH}"
  --sckite_vocab_path "${SCKITE_VOCAB_PATH}"
  --sckite_model_path "${SCKITE_MODEL_PATH}"
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

if [[ -n "${SCKITE_MODEL_KWARGS_JSON}" ]]; then
  ARGS+=(--sckite_model_kwargs_json "${SCKITE_MODEL_KWARGS_JSON}")
fi

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
