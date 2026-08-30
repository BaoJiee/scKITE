#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Project
# ============================================================

PROJECT_DIR=/root/autodl-tmp/ExpertCoder/ExpertCoder_Separate_decoder_activate_regulon

cd "${PROJECT_DIR}"


# ============================================================
# Conda
# ============================================================

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cllm


# ============================================================
# GPU / DDP
# ============================================================

# 使用两张 GPU
export CUDA_VISIBLE_DEVICES=0,1

# torchrun 启动的进程数，应与可见 GPU 数一致
NPROC_PER_NODE=2


# ============================================================
# Runtime environment
# ============================================================

export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export PYTHONUNBUFFERED=1

# 减少动态长度 batch 造成的 CUDA 显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ============================================================
# Paths
# ============================================================

# 当前 Dual-Decoder Stage2 配置文件
CONFIG_PATH="${PROJECT_DIR}/stage2.yaml"

# 当前统一训练脚本
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"

# 日志目录
LOG_DIR="${PROJECT_DIR}/logs"


# ============================================================
# Checkpoint options
# ============================================================

# ------------------------------------------------------------
# 1. RESUME
# ------------------------------------------------------------
# 用于恢复“当前双 Decoder Stage2 模型”的完整训练状态：
# model + optimizer + scheduler/scaler + epoch 等。
#
# 仅用于由当前 model.py 训练产生的 Stage2 checkpoint。
#
# 例如：
# RESUME="/root/autodl-tmp/ExpertCoder/.../runs/xxx/last.pt"
#
RESUME=""


# ------------------------------------------------------------
# 2. INIT_FROM
# ------------------------------------------------------------
# 仅按“参数名相同且 shape 相同”初始化模型权重，
# 不恢复 optimizer 和训练进度。
#
# 一般情况下，Stage1 encoder checkpoint 推荐直接写在 YAML：
#
# model:
#   stage1_ckpt_path: /path/to/stage1/best.pt
#
# 因此正常 Stage2 训练时这里保持为空即可。
#
# 如果需要从另一个兼容模型权重初始化，可填写：
# INIT_FROM="/path/to/model_checkpoint.pt"
#
INIT_FROM=""


# ============================================================
# Auto shutdown
# ============================================================

# true  : 训练成功后等待 60 秒并关机
# false : 训练成功后不关机
AUTO_SHUTDOWN=true

SHUTDOWN_DELAY=60


# ============================================================
# Create directories
# ============================================================

mkdir -p "${LOG_DIR}"


# ============================================================
# File checks
# ============================================================

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: CONFIG_PATH 不存在：${CONFIG_PATH}"
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: TRAIN_SCRIPT 不存在：${TRAIN_SCRIPT}"
    exit 1
fi


# RESUME 与 INIT_FROM 不建议同时使用
if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
    echo "ERROR: RESUME 和 INIT_FROM 不能同时设置。"
    echo "       RESUME 用于完整断点续训；INIT_FROM 仅用于权重初始化。"
    exit 1
fi

if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
    echo "ERROR: RESUME checkpoint 不存在：${RESUME}"
    exit 1
fi

if [[ -n "${INIT_FROM}" && ! -f "${INIT_FROM}" ]]; then
    echo "ERROR: INIT_FROM checkpoint 不存在：${INIT_FROM}"
    exit 1
fi

# ============================================================
# Verify no-Activity-Head / no-Test source code
# ============================================================

FORBIDDEN_PATTERNS=(
    "regulon_activity_head"
    "lambda_regulon_activity"
    "val_activity_macro_auprc"
    "val_activity_recall_at_3"
    "test_loader"
    "test_local"
    "evaluate_end2end_inference"
)

SOURCE_FILES=(
    "${PROJECT_DIR}/model.py"
    "${PROJECT_DIR}/data.py"
    "${PROJECT_DIR}/train.py"
)

for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    if grep -nH "${pattern}" "${SOURCE_FILES[@]}"; then
        echo "ERROR: 当前代码中仍检测到旧 Activity Head/Test 内容：${pattern}"
        echo "请确认使用的是无 Activity Head、无 Test 的新版脚本。"
        exit 1
    fi
done

echo "Source-code check passed: no Activity Head / no Test."

# ============================================================
# Validate YAML configuration and referenced paths
# ============================================================

python - "${CONFIG_PATH}" <<'PY'
import os
import sys
import yaml

config_path = sys.argv[1]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise SystemExit("ERROR: YAML 根对象不是字典。")

paths = cfg.get("paths", {})
model = cfg.get("model", {})
train = cfg.get("train", {})
data = cfg.get("data", {})

required_dirs = {
    "paths.train_local": paths.get("train_local"),
    "paths.val_local": paths.get("val_local"),
}

required_files = {
    "paths.regulon_target_path":
        paths.get("regulon_target_path"),
    "model.stage1_ckpt_path":
        model.get("stage1_ckpt_path"),
}

for name, path in required_dirs.items():
    if not path:
        raise SystemExit(f"ERROR: {name} 未配置。")
    if not os.path.isdir(path):
        raise SystemExit(
            f"ERROR: {name} 目录不存在：{path}"
        )

for name, path in required_files.items():
    if not path:
        raise SystemExit(f"ERROR: {name} 未配置。")
    if not os.path.isfile(path):
        raise SystemExit(
            f"ERROR: {name} 文件不存在：{path}"
        )

for forbidden in (
    "test_local",
    "test_path",
):
    if forbidden in paths:
        raise SystemExit(
            f"ERROR: 无 Test 版本不应包含 paths.{forbidden}"
        )

for forbidden in (
    "lambda_regulon_activity",
    "run_end2end_test",
    "test_end2end_max_new_tokens",
    "test_end2end_max_cells",
):
    if forbidden in train:
        raise SystemExit(
            f"ERROR: 无 Head/Test 版本不应包含 train.{forbidden}"
        )

expected_losses = {
    "lambda_expr": 1.0,
    "lambda_regulon": 1.0,
    "lambda_annotation": 1.0,
}

for key, expected in expected_losses.items():
    actual = float(train.get(key, float("nan")))
    if actual != expected:
        raise SystemExit(
            f"ERROR: train.{key}={actual}，预期为 {expected}"
        )

if int(data.get("regulon_num_queries", -1)) != 3:
    raise SystemExit(
        "ERROR: regulon_num_queries 必须为 3。"
    )

if str(data.get("task_sampling_mode", "")).lower() != "all":
    raise SystemExit(
        "ERROR: task_sampling_mode 应为 all。"
    )

print("YAML and path checks passed.")
print(f"train_local: {paths['train_local']}")
print(f"val_local: {paths['val_local']}")
print(
    "stage1_ckpt_path: "
    f"{model['stage1_ckpt_path']}"
)
PY
# ============================================================
# Build training arguments
# ============================================================

RUN_ARGS=(
    "${TRAIN_SCRIPT}"
    --config "${CONFIG_PATH}"
)

# 完整断点续训
if [[ -n "${RESUME}" ]]; then
    RUN_ARGS+=(
        --resume "${RESUME}"
    )
fi

# 仅初始化兼容权重
if [[ -n "${INIT_FROM}" ]]; then
    RUN_ARGS+=(
        --init_from "${INIT_FROM}"
    )
fi


# ============================================================
# Log
# ============================================================

LOG_FILE="${LOG_DIR}/stage2_dual_decoder_$(date +%Y%m%d_%H%M%S).log"


echo "============================================================" | tee -a "${LOG_FILE}"
echo "Stage2 Dual-Decoder Training" | tee -a "${LOG_FILE}"
echo "Shared Encoder + Regulon Decoder + Annotation Decoder" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

echo "Training started at: $(date)" | tee -a "${LOG_FILE}"

echo "PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}"
echo "CONFIG_PATH=${CONFIG_PATH}" | tee -a "${LOG_FILE}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}" | tee -a "${LOG_FILE}"

echo "RESUME=${RESUME:-<none>}" | tee -a "${LOG_FILE}"
echo "INIT_FROM=${INIT_FROM:-<none>}" | tee -a "${LOG_FILE}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}" | tee -a "${LOG_FILE}"

echo "AUTO_SHUTDOWN=${AUTO_SHUTDOWN}" | tee -a "${LOG_FILE}"
echo "LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "============================================================" | tee -a "${LOG_FILE}"


# ============================================================
# DDP Training
# ============================================================

if torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${RUN_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
then

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "Training finished successfully at $(date)" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    if [[ "${AUTO_SHUTDOWN}" == "true" ]]; then
        echo "Server will shutdown in ${SHUTDOWN_DELAY} seconds..." | tee -a "${LOG_FILE}"

        sync
        sleep "${SHUTDOWN_DELAY}"

        shutdown -h now || poweroff || halt
    else
        echo "AUTO_SHUTDOWN=false, server will remain running." | tee -a "${LOG_FILE}"
    fi

else

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "Training failed at $(date)" | tee -a "${LOG_FILE}"
    echo "Debug mode: server will NOT shutdown." | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    exit 1

fi
