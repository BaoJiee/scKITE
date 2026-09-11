#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/Gene_Perturbation_Prediction"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_scfoundation.py"
PYTHON_BIN="/root/miniconda3/envs/cllm/bin/python"

DATA_ROOT="${PROJECT_ROOT}/data"
DATASET_DIR="${DATA_ROOT}/norman_custom"
SCFOUNDATION_SOURCE_DIR="/root/autodl-tmp/other_model/scFoundation"
SCFOUNDATION_CKPT="/root/autodl-tmp/pt/scfoundation/models.ckpt"
CANONICAL_GENE_PATH="${SCFOUNDATION_SOURCE_DIR}/model/OS_scRNA_gene_index.19264.tsv"
HGNC_MAPPING_PATH="${PROJECT_ROOT}/data/gene_mapping/hgnc_complete_set.txt"
HGNC_SHA256="2106d1f237d6c542a85a4c399225e011ee9c5822199e7a400b7b9f842c8c8ca0"
SAVE_ROOT="${PROJECT_ROOT}/results"

# Fixed scFoundation input contract for this dataset:
# adata.X is already normalized+log1p, so use official pre_normalized=T.
PRE_NORMALIZED="T"
TARGET_HIGH_RESOLUTION="4"
CHECKPOINT_KEY="gene"
GENE_EMBEDDING_MODE="scfm_contextual"
PERT_EMBEDDING_MODE="scfm_init"
FREEZE_PERT_EMB="false"
CONTEXTUAL_FALLBACK="static"
CONTEXTUAL_ENCODER_BATCH_SIZE="1"
MISSING_STRATEGY="mean_gene"
SCFOUNDATION_PRECISION="fp16"
VERIFY_LOADED_TENSORS="true"
REQUIRE_ALL_PERTURBATIONS="true"

CUDA_DEVICE="0"
SEED="1"
SPLIT="simulation"
TRAIN_GENE_SET_SIZE="0.75"
COMBO_SEEN2_TRAIN_FRAC="0.75"

EPOCHS="15"
BATCH_SIZE="400"
TEST_BATCH_SIZE="400"
LR="1e-3"
WEIGHT_DECAY="5e-4"

# The selected scFoundation checkpoint returns 512-dimensional gene embeddings.
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
WANDB_PROJECT="GEARS_scFoundation"
WANDB_MODE="offline"
SHUTDOWN_AFTER_FINISH="ture"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export WANDB_MODE="${WANDB_MODE}"

if (( BATCH_SIZE > 8 || TEST_BATCH_SIZE > 8 )); then
  echo "Warning: scFoundation contextual mode decodes 19,264 genes per cell." >&2
  echo "Start with BATCH_SIZE=2 and TEST_BATCH_SIZE=2 before increasing them." >&2
fi

cd "${PROJECT_ROOT}"

for required_path in \
  "${TRAIN_SCRIPT}" \
  "${SCFOUNDATION_SOURCE_DIR}/model/load.py" \
  "${SCFOUNDATION_CKPT}" \
  "${CANONICAL_GENE_PATH}" \
  "${HGNC_MAPPING_PATH}" \
  "${DATASET_DIR}/perturb_processed.h5ad"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="scfoundation_norman_T_t${TARGET_HIGH_RESOLUTION}_pert${PERT_EMBEDDING_MODE}_hs${HIDDEN_SIZE}_seed${SEED}_${TIMESTAMP}"
SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"
LOG_DIR="${SAVE_DIR}/logs"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}"

echo "Run name: ${RUN_NAME}"
echo "scFoundation checkpoint: ${SCFOUNDATION_CKPT}"
echo "Input contract: pre_normalized=${PRE_NORMALIZED}, target=t${TARGET_HIGH_RESOLUTION}"
echo "GEARS batch: ${BATCH_SIZE}; scFoundation encoder batch: ${CONTEXTUAL_ENCODER_BATCH_SIZE}"
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
  --scfoundation_source_dir "${SCFOUNDATION_SOURCE_DIR}"
  --scfoundation_ckpt "${SCFOUNDATION_CKPT}"
  --canonical_gene_path "${CANONICAL_GENE_PATH}"
  --hgnc_mapping_path "${HGNC_MAPPING_PATH}"
  --hgnc_sha256 "${HGNC_SHA256}"
  --checkpoint_key "${CHECKPOINT_KEY}"
  --pre_normalized "${PRE_NORMALIZED}"
  --target_high_resolution "${TARGET_HIGH_RESOLUTION}"
  --gene_embedding_mode "${GENE_EMBEDDING_MODE}"
  --pert_embedding_mode "${PERT_EMBEDDING_MODE}"
  --freeze_pert_emb "${FREEZE_PERT_EMB}"
  --contextual_fallback "${CONTEXTUAL_FALLBACK}"
  --contextual_encoder_batch_size "${CONTEXTUAL_ENCODER_BATCH_SIZE}"
  --missing_strategy "${MISSING_STRATEGY}"
  --scfoundation_precision "${SCFOUNDATION_PRECISION}"
  --verify_loaded_tensors "${VERIFY_LOADED_TENSORS}"
  --require_all_perturbations "${REQUIRE_ALL_PERTURBATIONS}"
  --wandb "${WANDB}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${RUN_NAME}"
)

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

if [[ "${SHUTDOWN_AFTER_FINISH}" == "true" ]]; then
  shutdown -h now
fi
