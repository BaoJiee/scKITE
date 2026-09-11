#!/usr/bin/env bash  # 使用 bash 解释器。 #

set -e  # 任意命令报错时立即退出。 #
set -o pipefail  # 管道中任意一步失败时立即退出。 #

PROJECT_ROOT="/root/autodl-tmp/Gene_Perturbation_Prediction"  # 设置项目根目录，也就是 gears 包所在目录的上一级。 #
TRAIN_SCRIPT="${PROJECT_ROOT}/train.py"  # 设置统一训练脚本路径。 #
PYTHON_BIN="python"  # 设置 Python 命令，如果需要可改成 /root/miniconda3/envs/你的环境/bin/python。 #

INPUT_H5AD="/root/autodl-tmp/Gene_Perturbation_Prediction/data/norman_custom/perturb_processed.h5ad"  # 你的 GEARS 格式 h5ad。 #
DATA_ROOT="/root/autodl-tmp/Gene_Perturbation_Prediction/data"  # GEARS 数据根目录。 #
DATASET_NAME="norman_custom"  # 新数据集名字。 #
DATASET_DIR="${DATA_ROOT}/${DATASET_NAME}"  # 训练脚本读取的数据集目录。 #
FORCE_REBUILD_DATA="false"  # 是否删除旧 data_pyg 和 splits 后重建。 #

SAVE_ROOT="/root/autodl-tmp/Gene_Perturbation_Prediction/results"  # 设置结果保存根目录。 #
STAGE2_CKPT_PATH="/root/autodl-tmp/pt/geosketch_25_encoderonly.pt"  # 设置 Stage2 best.pt 路径。 #
STAGE2_VOCAB_PATH="/root/autodl-tmp/vocab/gene_vocabulary.jsonl"  # 设置 Stage2 gene_vocabulary.jsonl 路径。 #
STAGE2_MODEL_PY_PATH="/root/autodl-tmp/ExpertCoder/ExpertCoder_Separate_decoder_activate_regulon/model.py"  # 设置 Stage2 模型结构文件路径，contextual 模式需要。 #
STAGE2_MODEL_KWARGS_JSON=""  # 设置 Stage2 模型初始化参数 JSON 或 JSON 文件路径，static 模式可为空。 #

SCFM_MODE="contextual"  # 设置 SCFM gene embedding 模式，可选 static 或 contextual。 #
PERT_MODE="native"  # 设置 perturbation embedding 模式，可选 scfm_init 或 native。 #
FREEZE_PERT_EMB="false"  # 设置外部 pert embedding 初始化后是否冻结。 #
PROJECT_METHOD="mean_pool"  # 设置 adapter 内部投影方式，可选 slice 或 mean_pool。 #
MISSING_STRATEGY="mean_gene"  # 设置缺失 gene/pert 的 embedding 填充方式，可选 mean_gene、mean_all、zero。 #
NORMALIZE_EMBEDDING="false"  # 设置是否对 adapter 输出 embedding 做 L2 normalize。 #



# contextual_value_mode 可选：
# bin   = 使用原来的 51-bin
# as_is = 关闭 bin，直接使用 cell_graphs.pkl 里的表达量
# log1p = 在 adapter 内部再做 log1p，一般不推荐，因为你的 adata.X 通常已经是 log1p
CONTEXTUAL_VALUE_MODE="as_is"
NUM_BINS="51"  # 设置 contextual 模式表达量分桶数量。 #

# contextual_max_genes：
# 1200 = 只让 1200 个 gene 进入 Stage2 encoder
# 0 或空值 = 不限制，全部 gene 进入 Stage2 encoder
CONTEXTUAL_MAX_GENES="0"

# contextual_gene_selection 可选：
# matched_first = 优先选择能匹配 Stage2 vocab 的 gene
# first         = 直接选 GEARS gene_list 前 K 个
CONTEXTUAL_GENE_SELECTION="matched_first"

# contextual_fallback 可选：
# static = 没进入 contextual encoder 的 gene 用 static scFM embedding 补
# zero   = 没进入 contextual encoder 的 gene 用 0 向量补
CONTEXTUAL_FALLBACK="static"

CUDA_DEVICE="0"  # 设置使用哪张 GPU。 #
SEED="1"  # 设置随机种子。 #
SPLIT="simulation"  # 设置 GEARS 数据划分方式。 #
TRAIN_GENE_SET_SIZE="0.75"  # 设置 simulation split 中训练扰动基因比例。 #
COMBO_SEEN2_TRAIN_FRAC="0.75"  # 设置 combo_seen2 进入训练集的比例。 #

EPOCHS="15"  # 设置训练轮数。 #
BATCH_SIZE="420"  # 设置训练 batch size；contextual 模式建议先设为 1。 #
TEST_BATCH_SIZE="400"  # 设置验证和测试 batch size；contextual 模式建议先设为 1。 #
LR="1e-4"  # 设置学习率。 #1e-3
WEIGHT_DECAY="5e-4"  # 设置 weight decay。 #

HIDDEN_SIZE="512"  # 设置 GEARS hidden_size，同时也是 Stage2Adapter 输出维度。 #
NUM_GO_GNN_LAYERS="1"  # 设置 GO graph GNN 层数。 #
NUM_GENE_GNN_LAYERS="1"  # 设置 co-expression graph GNN 层数。 #
DECODER_HIDDEN_SIZE="16"  # 设置 gene-specific decoder hidden size。 #
NUM_SIMILAR_GENES_GO_GRAPH="20"  # 设置 GO graph 每个 target 保留的相似基因数。 #
NUM_SIMILAR_GENES_COEXPRESS_GRAPH="20"  # 设置 co-expression graph 每个 target 保留的相似基因数。 #
COEXPRESS_THRESHOLD="0.4"  # 设置共表达网络相关性阈值。 #
UNCERTAINTY="false"  # 设置是否开启 uncertainty 模式。 #
UNCERTAINTY_REG="1"  # 设置 uncertainty 正则权重。 #
DIRECTION_LAMBDA="1e-1"  # 设置 direction loss 权重。 #

WANDB="true"  # 设置是否启用 wandb，true 或 false。 #
WANDB_PROJECT="run_scfm_gears"  # 设置 wandb project 名称。 #
WANDB_MODE="online"  # 设置 wandb 模式，可选 online、offline、disabled。 #

SHUTDOWN_AFTER_FINISH="true"  # 设置训练结束后是否自动关机，true 或 false。 #

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"  # 导出 GPU 设置。 #
export TOKENIZERS_PARALLELISM=false  # 关闭 tokenizer 并行警告。 #
export OMP_NUM_THREADS=4  # 设置 OpenMP 线程数。 #
export MKL_NUM_THREADS=4  # 设置 MKL 线程数。 #
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 设置 PyTorch CUDA 显存分配策略。 #
export CUDA_MODULE_LOADING=LAZY
export WANDB_MODE="${WANDB_MODE}"  # 导出 wandb 模式。 #

cd "${PROJECT_ROOT}"  # 进入项目根目录。 #

if [[ "${SCFM_MODE}" == "static" ]]; then  # 判断是否使用 static gene embedding。 #
  GENE_EMBEDDING_MODE="scfm_static"  # 设置 GEARS gene embedding 模式为 scfm_static。 #
elif [[ "${SCFM_MODE}" == "contextual" ]]; then  # 判断是否使用 contextual gene embedding。 #
  GENE_EMBEDDING_MODE="scfm_contextual"  # 设置 GEARS gene embedding 模式为 scfm_contextual。 #
else  # 如果 SCFM_MODE 不合法。 #
  echo "Invalid SCFM_MODE=${SCFM_MODE}, please use static or contextual."  # 打印错误信息。 #
  exit 1  # 退出脚本。 #
fi  # 结束 SCFM_MODE 判断。 #

if [[ "${SCFM_MODE}" == "contextual" ]]; then  # 如果使用 contextual 模式。 #
  echo "[Warning] contextual mode is memory-heavy; BATCH_SIZE=1 is recommended for first test."  # 打印显存提醒。 #
fi  # 结束 contextual 提醒。 #

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"  # 生成当前时间戳。 #
RUN_NAME="geosketch_25_encoderonly_${DATASET_NAME}_${SCFM_MODE}_${CONTEXTUAL_VALUE_MODE}_${CONTEXTUAL_MAX_GENES}_pert${PERT_MODE}_hs${HIDDEN_SIZE}_seed${SEED}_${TIMESTAMP}"  # 设置本次运行名称。 #
SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"  # 设置本次模型保存目录。 #
LOG_DIR="${SAVE_DIR}/logs"  # 设置日志目录。 #
LOG_FILE="${LOG_DIR}/train.log"  # 设置日志文件路径。 #

mkdir -p "${LOG_DIR}"  # 创建日志目录。 #

echo "Run name: ${RUN_NAME}"  # 打印运行名称。 #
echo "SCFM mode: ${SCFM_MODE}"  # 打印 SCFM 模式。 #
echo "Gene embedding mode: ${GENE_EMBEDDING_MODE}"  # 打印 gene embedding 模式。 #
echo "Perturbation mode: ${PERT_MODE}"  # 打印 perturbation embedding 模式。 #
echo "Save dir: ${SAVE_DIR}"  # 打印保存目录。 #
echo "Log file: ${LOG_FILE}"  # 打印日志文件。 #

ARGS=(  # 开始构建训练参数数组。 #
  --data_root "${DATA_ROOT}"  # 传入 GEARS 数据根目录。 #
  --dataset_dir "${DATASET_DIR}"  # 传入数据集目录。 #
  --save_dir "${SAVE_DIR}"  # 传入模型保存目录。 #
  --device cuda  # 设置训练设备。 #
  --seed "${SEED}"  # 传入随机种子。 #
  --split "${SPLIT}"  # 传入 split 类型。 #
  --train_gene_set_size "${TRAIN_GENE_SET_SIZE}"  # 传入训练扰动基因比例。 #
  --combo_seen2_train_frac "${COMBO_SEEN2_TRAIN_FRAC}"  # 传入 combo_seen2 训练比例。 #
  --batch_size "${BATCH_SIZE}"  # 传入训练 batch size。 #
  --test_batch_size "${TEST_BATCH_SIZE}"  # 传入测试 batch size。 #
  --epochs "${EPOCHS}"  # 传入训练轮数。 #
  --lr "${LR}"  # 传入学习率。 #
  --weight_decay "${WEIGHT_DECAY}"  # 传入 weight decay。 #
  --hidden_size "${HIDDEN_SIZE}"  # 传入 GEARS hidden_size。 #
  --num_go_gnn_layers "${NUM_GO_GNN_LAYERS}"  # 传入 GO GNN 层数。 #
  --num_gene_gnn_layers "${NUM_GENE_GNN_LAYERS}"  # 传入 gene GNN 层数。 #
  --decoder_hidden_size "${DECODER_HIDDEN_SIZE}"  # 传入 decoder hidden size。 #
  --num_similar_genes_go_graph "${NUM_SIMILAR_GENES_GO_GRAPH}"  # 传入 GO 图 K。 #
  --num_similar_genes_co_express_graph "${NUM_SIMILAR_GENES_COEXPRESS_GRAPH}"  # 传入共表达图 K。 #
  --coexpress_threshold "${COEXPRESS_THRESHOLD}"  # 传入共表达阈值。 #
  --uncertainty "${UNCERTAINTY}"  # 传入 uncertainty 设置。 #
  --uncertainty_reg "${UNCERTAINTY_REG}"  # 传入 uncertainty 正则权重。 #
  --direction_lambda "${DIRECTION_LAMBDA}"  # 传入方向损失权重。 #
  --no_perturb false  # 关闭 no perturb 模式。 #
  --adapter_name stage2  # 使用 Stage2Adapter。 #
  --stage2_ckpt_path "${STAGE2_CKPT_PATH}"  # 传入 Stage2 checkpoint 路径。 #
  --stage2_vocab_path "${STAGE2_VOCAB_PATH}"  # 传入 Stage2 vocab 路径。 #
  --stage2_model_py_path "${STAGE2_MODEL_PY_PATH}"  # 传入 Stage2 模型结构文件路径。 #
  --num_bins "${NUM_BINS}"  # 传入表达量分桶数量。 #
  --project_method "${PROJECT_METHOD}"  # 传入 adapter 内投影方式。 #
  --missing_strategy "${MISSING_STRATEGY}"  # 传入缺失 embedding 策略。 #
  --normalize_embedding "${NORMALIZE_EMBEDDING}"  # 传入是否归一化 embedding。 #
  --gene_embedding_mode "${GENE_EMBEDDING_MODE}"  # 传入 gene embedding 模式。 #
  --pert_embedding_mode "${PERT_MODE}"  # 传入 perturbation embedding 模式。 #
  --freeze_pert_emb "${FREEZE_PERT_EMB}"  # 传入是否冻结 pert_emb。 #
  --wandb "${WANDB}"  # 传入是否启用 wandb。 #
  --wandb_project "${WANDB_PROJECT}"  # 传入 wandb project 名称。 #
  --wandb_run_name "${RUN_NAME}"  # 传入 wandb run 名称。 #
  --contextual_value_mode "${CONTEXTUAL_VALUE_MODE}"
  --contextual_max_genes "${CONTEXTUAL_MAX_GENES}"
  --contextual_gene_selection "${CONTEXTUAL_GENE_SELECTION}"
  --contextual_fallback "${CONTEXTUAL_FALLBACK}"
)  # 训练参数数组结束。 #

if [[ -n "${STAGE2_MODEL_KWARGS_JSON}" ]]; then  # 如果提供了 Stage2 模型参数 JSON。 #
  ARGS+=(--stage2_model_kwargs_json "${STAGE2_MODEL_KWARGS_JSON}")  # 追加 Stage2 模型参数 JSON。 #
fi  # 结束 Stage2 模型参数判断。 #

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"  # 执行训练并保存日志。 #

if [[ "${SHUTDOWN_AFTER_FINISH}" == "true" ]]; then  # 判断是否需要训练结束后关机。 #
  shutdown -h now  # 训练完成后关机。 #
fi  # 结束关机判断。 #
