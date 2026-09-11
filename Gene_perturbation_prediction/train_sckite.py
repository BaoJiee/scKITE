#!/usr/bin/env python3  # 指定 Python3 解释器。 #
# -*- coding: utf-8 -*-  # 指定 UTF-8 编码。 #

import argparse  # 导入命令行参数解析模块。 #
import json  # 导入 json 模块。 #
import os  # 导入系统路径模块。 #
import random  # 导入随机数模块。 #
from pathlib import Path  # 导入 Path 路径工具。 #

import numpy as np  # 导入 numpy。 #
import torch  # 导入 PyTorch。 #

from gears import PertData, GEARS  # 导入 GEARS 和 PertData。 #


def str2bool(v):  # 定义字符串转 bool 的函数。 #
    if isinstance(v, bool):  # 如果输入本身是 bool。 #
        return v  # 直接返回。 #
    v = str(v).lower()  # 转为小写字符串。 #
    if v in ["yes", "true", "t", "1", "y"]:  # 判断 True 值。 #
        return True  # 返回 True。 #
    if v in ["no", "false", "f", "0", "n"]:  # 判断 False 值。 #
        return False  # 返回 False。 #
    raise argparse.ArgumentTypeError("Boolean value expected.")  # 非法 bool 参数时报错。 #


def set_seed(seed):  # 定义随机种子设置函数。 #
    random.seed(seed)  # 设置 Python 随机种子。 #
    np.random.seed(seed)  # 设置 numpy 随机种子。 #
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机种子。 #
    if torch.cuda.is_available():  # 如果 CUDA 可用。 #
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子。 #


def load_json_arg(value):  # 定义读取 JSON 参数的函数。 #
    if value is None or str(value).strip() == "":  # 如果参数为空。 #
        return None  # 返回 None。 #
    value = str(value)  # 转成字符串。 #
    if Path(value).exists():  # 如果参数是文件路径。 #
        with open(value, "r", encoding="utf-8") as f:  # 打开 JSON 文件。 #
            return json.load(f)  # 读取 JSON 内容。 #
    return json.loads(value)  # 否则把参数当 JSON 字符串解析。 #


def build_adapter_kwargs(args, device):  # 根据 adapter_name 构建 adapter_kwargs。 #
    if args.adapter_name == "native_gears":  # 如果使用原始 GEARS。 #
        return {}  # native 模式不需要 adapter，但传空字典避免 dict(None) 报错。 #

    if args.adapter_name == "random":  # 如果使用 random adapter。 #
        return {  # 返回 random adapter 参数。 #
            "device": device,  # 设置设备。 #
            "output_dim": args.hidden_size,  # random adapter 输出维度等于 GEARS hidden_size。 #
            "seed": args.seed,  # 设置随机种子。 #
        }  # random adapter 参数结束。 #

    if args.adapter_name == "stage2":  # 如果使用 Stage2 adapter。 #
        if args.stage2_ckpt_path is None:  # 检查 checkpoint 路径。 #
            raise ValueError("--stage2_ckpt_path is required when adapter_name='stage2'.")  # 缺失则报错。 #
        if args.stage2_vocab_path is None:  # 检查 vocab 路径。 #
            raise ValueError("--stage2_vocab_path is required when adapter_name='stage2'.")  # 缺失则报错。 #

        adapter_kwargs = {
            "ckpt_path": args.stage2_ckpt_path,
            "vocab_path": args.stage2_vocab_path,
            "model_py_path": args.stage2_model_py_path,
            "gears_hidden_size": args.hidden_size,
            "num_bins": args.num_bins,
            "contextual_value_mode": args.contextual_value_mode,
            "contextual_max_genes": None if args.contextual_max_genes <= 0 else args.contextual_max_genes,
            "contextual_gene_selection": args.contextual_gene_selection,
            "contextual_fallback": args.contextual_fallback,
        }

        if args.gene_embedding_mode == "scfm_contextual":  # 如果使用 contextual gene embedding。 #
            if args.stage2_model_py_path is None:  # 检查模型结构文件。 #
                raise ValueError("--stage2_model_py_path is required when gene_embedding_mode='scfm_contextual'.")  # 缺失则报错。 #
            adapter_kwargs["model_py_path"] = args.stage2_model_py_path  # 添加 Stage2 模型结构文件路径。 #
            adapter_kwargs["num_bins"] = args.num_bins  # 添加表达量分桶数量。 #
            model_kwargs = load_json_arg(args.stage2_model_kwargs_json)  # 读取模型初始化参数。 #
            if model_kwargs is not None:  # 如果用户提供了模型参数。 #
                adapter_kwargs["model_kwargs"] = model_kwargs  # 写入 model_kwargs。 #

        return adapter_kwargs  # 返回 Stage2Adapter 参数。 #

    raise ValueError(f"Unknown adapter_name: {args.adapter_name}")  # 未知 adapter 报错。 #


def parse_args():  # 定义命令行参数解析函数。 #
    parser = argparse.ArgumentParser(description="Unified GEARS / SCFM-GEARS training script.")  # 创建参数解析器。 #

    parser.add_argument("--data_root", type=str, required=True, help="GEARS data root directory.")  # 数据根目录。 #
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory containing perturb_processed.h5ad.")  # 数据集目录。 #
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save trained GEARS model.")  # 模型保存目录。 #
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu.")  # 设备。 #
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")  # 随机种子。 #

    parser.add_argument("--split", type=str, default="simulation", help="GEARS split type.")  # split 类型。 #
    parser.add_argument("--train_gene_set_size", type=float, default=0.75, help="Train gene set size for simulation split.")  # simulation split 训练基因比例。 #
    parser.add_argument("--combo_seen2_train_frac", type=float, default=0.75, help="Combo seen2 train fraction.")  # combo seen2 训练比例。 #
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")  # 训练 batch size。 #
    parser.add_argument("--test_batch_size", type=int, default=8, help="Validation/test batch size.")  # 测试 batch size。 #

    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")  # 训练轮数。 #
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")  # 学习率。 #
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")  # 权重衰减。 #

    parser.add_argument("--hidden_size", type=int, default=64, help="GEARS hidden size.")  # GEARS hidden_size。 #
    parser.add_argument("--num_go_gnn_layers", type=int, default=1, help="Number of GO GNN layers.")  # GO GNN 层数。 #
    parser.add_argument("--num_gene_gnn_layers", type=int, default=1, help="Number of co-expression GNN layers.")  # gene GNN 层数。 #
    parser.add_argument("--decoder_hidden_size", type=int, default=16, help="Gene-specific decoder hidden size.")  # decoder hidden size。 #
    parser.add_argument("--num_similar_genes_go_graph", type=int, default=20, help="K for GO graph.")  # GO 图 K。 #
    parser.add_argument("--num_similar_genes_co_express_graph", type=int, default=20, help="K for co-expression graph.")  # 共表达图 K。 #
    parser.add_argument("--coexpress_threshold", type=float, default=0.4, help="Co-expression threshold.")  # 共表达阈值。 #
    parser.add_argument("--uncertainty", type=str2bool, default=False, help="Whether to use uncertainty mode.")  # 不确定性模式。 #
    parser.add_argument("--uncertainty_reg", type=float, default=1.0, help="Uncertainty regularization.")  # 不确定性正则。 #
    parser.add_argument("--direction_lambda", type=float, default=1e-1, help="Direction loss weight.")  # 方向损失权重。 #
    parser.add_argument("--no_perturb", type=str2bool, default=False, help="Predict no perturbation condition.")  # no perturb 模式。 #

    parser.add_argument("--adapter_name", type=str, default="native_gears", choices=["native_gears", "random", "stage2"], help="Adapter name.")  # adapter 名称。 #
    parser.add_argument("--gene_embedding_mode", type=str, default="native", choices=["native", "scfm_static", "scfm_contextual"], help="Gene embedding mode.")  # gene embedding 模式。 #
    parser.add_argument("--pert_embedding_mode", type=str, default="native", choices=["native", "scfm_init"], help="Perturbation embedding mode.")  # pert embedding 模式。 #
    parser.add_argument("--freeze_pert_emb", type=str2bool, default=False, help="Freeze pert_emb after SCFM init.")  # 是否冻结 pert_emb。 #

    parser.add_argument("--stage2_ckpt_path", type=str, default=None, help="Path to Stage2 best.pt.")  # Stage2 checkpoint 路径。 #
    parser.add_argument("--stage2_vocab_path", type=str, default=None, help="Path to gene_vocabulary.jsonl.")  # Stage2 vocab 路径。 #
    parser.add_argument("--stage2_model_py_path", type=str, default=None, help="Path to model_stage2_mixed.py for contextual mode.")  # Stage2 模型结构文件路径。 #
    parser.add_argument("--stage2_model_kwargs_json", type=str, default=None, help="JSON string or JSON file for TahoeStage2MixedModel kwargs.")  # Stage2 模型参数 JSON。 #
    parser.add_argument("--num_bins", type=int, default=51, help="Expression bins for contextual Stage2 encoder.")  # 表达量 bin 数。 #
    parser.add_argument("--project_method", type=str, default="slice", choices=["slice", "mean_pool"], help="Projection method inside adapter.")  # adapter 内投影方法。 #
    parser.add_argument("--missing_strategy", type=str, default="mean_gene", choices=["mean_gene", "mean_all", "zero"], help="Missing gene embedding strategy.")  # 缺失基因策略。 #
    parser.add_argument("--normalize_embedding", type=str2bool, default=False, help="Whether to L2 normalize adapter embeddings.")  # 是否归一化。 #
    parser.add_argument("--contextual_value_mode", type=str, default="bin")  # bin / as_is / log1p。 #
    parser.add_argument("--contextual_max_genes", type=int, default=0)  # 0 表示不限制，1200 表示只送 1200 个 gene。 #
    parser.add_argument("--contextual_gene_selection", type=str, default="matched_first")  # matched_first / first。 #
    parser.add_argument("--contextual_fallback", type=str, default="static")  # static / zero。 #

    parser.add_argument("--wandb", type=str2bool, default=False, help="Whether to use wandb.")  # 是否启用 wandb。 #
    parser.add_argument("--wandb_project", type=str, default="GEARS", help="Wandb project name.")  # wandb 项目名。 #
    parser.add_argument("--wandb_run_name", type=str, default="GEARS_adapter_run", help="Wandb run name.")  # wandb run 名。 #

    parser.add_argument("--save_run_config", type=str2bool, default=True, help="Whether to save run_config.json.")  # 是否保存运行配置。 #
    return parser.parse_args()  # 返回解析结果。 #


def main():  # 定义主函数。 #
    args = parse_args()  # 解析参数。 #
    set_seed(args.seed)  # 设置随机种子。 #

    device = args.device  # 读取设备。 #
    if device == "cuda" and not torch.cuda.is_available():  # 如果要求 cuda 但不可用。 #
        device = "cpu"  # 回退到 CPU。 #
        print("[Warning] CUDA is not available. Use CPU instead.", flush=True)  # 打印警告。 #

    save_dir = Path(args.save_dir)  # 构建保存目录 Path。 #
    save_dir.mkdir(parents=True, exist_ok=True)  # 创建保存目录。 #

    if args.save_run_config:  # 如果需要保存运行配置。 #
        with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:  # 打开配置文件。 #
            json.dump(vars(args), f, indent=2, ensure_ascii=False)  # 保存参数。 #

    print("========== GEARS Adapter Training ==========", flush=True)  # 打印标题。 #
    print(f"device: {device}", flush=True)  # 打印设备。 #
    print(f"adapter_name: {args.adapter_name}", flush=True)  # 打印 adapter 名称。 #
    print(f"gene_embedding_mode: {args.gene_embedding_mode}", flush=True)  # 打印 gene embedding 模式。 #
    print(f"pert_embedding_mode: {args.pert_embedding_mode}", flush=True)  # 打印 pert embedding 模式。 #
    print(f"hidden_size: {args.hidden_size}", flush=True)  # 打印 hidden size。 #
    print(f"save_dir: {save_dir}", flush=True)  # 打印保存目录。 #

    print("Step 1: Load PertData.", flush=True)  # 打印步骤。 #
    pert_data = PertData(data_path=args.data_root)  # 初始化 PertData。 #
    pert_data.load(data_path=args.dataset_dir)  # 加载自定义数据集目录。 #

    print("Step 2: Prepare split.", flush=True)  # 打印步骤。 #
    pert_data.prepare_split(  # 准备 split。 #
        split=args.split,  # split 类型。 #
        seed=args.seed,  # 随机种子。 #
        train_gene_set_size=args.train_gene_set_size,  # 训练基因比例。 #
        combo_seen2_train_frac=args.combo_seen2_train_frac,  # combo seen2 训练比例。 #
    )  # split 准备结束。 #

    print("Step 3: Build dataloaders.", flush=True)  # 打印步骤。 #
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)  # 构建 dataloader。 #

    print("Step 4: Build adapter kwargs.", flush=True)  # 打印步骤。 #
    adapter_kwargs = build_adapter_kwargs(args, device=device)  # 构建 adapter 参数。 #

    print("Step 5: Initialize GEARS.", flush=True)  # 打印步骤。 #
    gears_model = GEARS(  # 初始化 GEARS 外层对象。 #
        pert_data,  # 传入 PertData。 #
        device=device,  # 设置设备。 #
        weight_bias_track=args.wandb,  # 是否启用 wandb。 #
        proj_name=args.wandb_project,  # wandb 项目名。 #
        exp_name=args.wandb_run_name,  # wandb run 名。 #
    )  # GEARS 初始化结束。 #

    gears_model.model_initialize(  # 初始化 GEARS 模型。 #
        hidden_size=args.hidden_size,  # GEARS hidden size。 #
        num_go_gnn_layers=args.num_go_gnn_layers,  # GO GNN 层数。 #
        num_gene_gnn_layers=args.num_gene_gnn_layers,  # gene GNN 层数。 #
        decoder_hidden_size=args.decoder_hidden_size,  # decoder hidden size。 #
        num_similar_genes_go_graph=args.num_similar_genes_go_graph,  # GO 图 K。 #
        num_similar_genes_co_express_graph=args.num_similar_genes_co_express_graph,  # 共表达图 K。 #
        coexpress_threshold=args.coexpress_threshold,  # 共表达阈值。 #
        uncertainty=args.uncertainty,  # 不确定性模式。 #
        uncertainty_reg=args.uncertainty_reg,  # 不确定性正则。 #
        direction_lambda=args.direction_lambda,  # 方向损失权重。 #
        no_perturb=args.no_perturb,  # no perturb 模式。 #
        adapter_name=args.adapter_name,  # adapter 名称。 #
        adapter_kwargs=adapter_kwargs,  # adapter 参数。 #
        gene_embedding_mode=args.gene_embedding_mode,  # gene embedding 模式。 #
        pert_embedding_mode=args.pert_embedding_mode,  # perturbation embedding 模式。 #
        freeze_pert_emb=args.freeze_pert_emb,  # 是否冻结 perturbation embedding。 #
    )  # 模型初始化结束。 #

    print("Step 6: Train GEARS.", flush=True)  # 打印步骤。 #
    gears_model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)  # 开始训练。 #

    print("Step 7: Save model.", flush=True)  # 打印步骤。 #
    gears_model.save_model(str(save_dir))  # 保存模型。 #

    print("Training finished.", flush=True)  # 打印训练结束。 #


if __name__ == "__main__":  # 如果作为脚本直接运行。 #
    main()  # 执行主函数。 #
