#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from adapters.scgpt_adapter import ScGPTAdapter
from gears import GEARS, PertData


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GEARS with a frozen scGPT embedding adapter."
    )

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--split", default="simulation")
    parser.add_argument("--train_gene_set_size", type=float, default=0.75)
    parser.add_argument("--combo_seen2_train_frac", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_go_gnn_layers", type=int, default=1)
    parser.add_argument("--num_gene_gnn_layers", type=int, default=1)
    parser.add_argument("--decoder_hidden_size", type=int, default=16)
    parser.add_argument("--num_similar_genes_go_graph", type=int, default=20)
    parser.add_argument(
        "--num_similar_genes_co_express_graph", type=int, default=20
    )
    parser.add_argument("--coexpress_threshold", type=float, default=0.4)
    parser.add_argument("--uncertainty", type=str2bool, default=False)
    parser.add_argument("--uncertainty_reg", type=float, default=1.0)
    parser.add_argument("--direction_lambda", type=float, default=1e-1)
    parser.add_argument("--no_perturb", type=str2bool, default=False)

    parser.add_argument("--scgpt_model_dir", required=True)
    parser.add_argument("--scgpt_source_dir", default=None)
    parser.add_argument(
        "--gene_embedding_mode",
        choices=["scfm_static", "scfm_contextual"],
        default="scfm_contextual",
    )
    parser.add_argument(
        "--pert_embedding_mode",
        choices=["native", "scfm_init"],
        default="scfm_init",
    )
    parser.add_argument("--freeze_pert_emb", type=str2bool, default=False)
    parser.add_argument(
        "--contextual_value_mode",
        choices=["bin", "as_is", "log1p"],
        default="bin",
    )
    parser.add_argument("--num_bins", type=int, default=51)
    parser.add_argument("--contextual_max_genes", type=int, default=1200)
    parser.add_argument(
        "--contextual_gene_selection",
        choices=["sample", "top_expression", "first", "matched_first"],
        default="sample",
    )
    parser.add_argument(
        "--contextual_fallback", choices=["static", "zero"], default="static"
    )
    parser.add_argument("--contextual_encoder_batch_size", type=int, default=8)
    parser.add_argument(
        "--missing_strategy",
        choices=["mean_gene", "mean_all", "zero"],
        default="mean_gene",
    )
    parser.add_argument(
        "--project_method", choices=["slice", "mean_pool"], default="slice"
    )
    parser.add_argument("--normalize_embedding", type=str2bool, default=False)
    parser.add_argument(
        "--scgpt_precision", choices=["fp32", "fp16", "bf16"], default="fp16"
    )

    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", default="GEARS_scGPT")
    parser.add_argument("--wandb_run_name", default="GEARS_scGPT_run")
    parser.add_argument("--save_run_config", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA is unavailable; falling back to CPU.", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_run_config:
        with open(save_dir / "run_config.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print("========== GEARS + frozen scGPT training ==========", flush=True)
    print(f"device: {device}", flush=True)
    print(f"gene_embedding_mode: {args.gene_embedding_mode}", flush=True)
    print(f"pert_embedding_mode: {args.pert_embedding_mode}", flush=True)
    print(f"contextual_value_mode: {args.contextual_value_mode}", flush=True)
    print(f"hidden_size: {args.hidden_size}", flush=True)
    print(f"save_dir: {save_dir}", flush=True)

    print("Step 1: Load PertData.", flush=True)
    pert_data = PertData(data_path=args.data_root)
    pert_data.load(data_path=args.dataset_dir)

    print("Step 2: Prepare split.", flush=True)
    pert_data.prepare_split(
        split=args.split,
        seed=args.seed,
        train_gene_set_size=args.train_gene_set_size,
        combo_seen2_train_frac=args.combo_seen2_train_frac,
    )

    print("Step 3: Build dataloaders.", flush=True)
    pert_data.get_dataloader(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )

    print("Step 4: Build frozen scGPT adapter.", flush=True)
    adapter_kwargs = {
        "model_dir": args.scgpt_model_dir,
        "scgpt_source_dir": args.scgpt_source_dir,
        "device": device,
        "gears_hidden_size": args.hidden_size,
        "max_seq_len": args.contextual_max_genes,
        "num_bins": args.num_bins,
        "contextual_value_mode": args.contextual_value_mode,
        "contextual_gene_selection": args.contextual_gene_selection,
        "contextual_fallback": args.contextual_fallback,
        "contextual_encoder_batch_size": args.contextual_encoder_batch_size,
        "missing_strategy": args.missing_strategy,
        "project_method": args.project_method,
        "normalize": args.normalize_embedding,
        "precision": args.scgpt_precision,
        "seed": args.seed,
    }
    adapter = ScGPTAdapter(**adapter_kwargs)
    with open(save_dir / "scgpt_adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump(adapter.get_config(), handle, indent=2, ensure_ascii=False)

    print("Step 5: Initialize GEARS.", flush=True)
    gears_model = GEARS(
        pert_data,
        device=device,
        weight_bias_track=args.wandb,
        proj_name=args.wandb_project,
        exp_name=args.wandb_run_name,
    )
    gears_model.model_initialize(
        hidden_size=args.hidden_size,
        num_go_gnn_layers=args.num_go_gnn_layers,
        num_gene_gnn_layers=args.num_gene_gnn_layers,
        decoder_hidden_size=args.decoder_hidden_size,
        num_similar_genes_go_graph=args.num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=(
            args.num_similar_genes_co_express_graph
        ),
        coexpress_threshold=args.coexpress_threshold,
        uncertainty=args.uncertainty,
        uncertainty_reg=args.uncertainty_reg,
        direction_lambda=args.direction_lambda,
        no_perturb=args.no_perturb,
        adapter=adapter,
        adapter_name="scgpt",
        adapter_kwargs=adapter_kwargs,
        gene_embedding_mode=args.gene_embedding_mode,
        pert_embedding_mode=args.pert_embedding_mode,
        freeze_pert_emb=args.freeze_pert_emb,
        num_bins=args.num_bins,
        contextual_value_mode=args.contextual_value_mode,
        contextual_max_genes=args.contextual_max_genes,
        contextual_gene_selection=args.contextual_gene_selection,
        contextual_fallback=args.contextual_fallback,
    )

    print("Step 6: Train GEARS.", flush=True)
    gears_model.train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Step 7: Save GEARS model.", flush=True)
    gears_model.save_model(str(save_dir))
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
