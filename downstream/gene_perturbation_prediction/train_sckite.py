#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from gears import PertData, GEARS


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ["yes", "true", "t", "1", "y"]:
        return True
    if v in ["no", "false", "f", "0", "n"]:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_arg(value):
    if value is None or str(value).strip() == "":
        return None
    value = str(value)
    if Path(value).exists():
        with open(value, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def build_adapter_kwargs(args, device):
    if args.adapter_name == "native_gears":
        return {}

    if args.adapter_name == "random":
        return {
            "device": device,
            "output_dim": args.hidden_size,
            "seed": args.seed,
        }

    if args.adapter_name == "sckite":
        if args.sckite_checkpoint_path is None:
            raise ValueError("--sckite_checkpoint_path is required when adapter_name='sckite'.")
        if args.sckite_vocab_path is None:
            raise ValueError("--sckite_vocab_path is required when adapter_name='sckite'.")

        adapter_kwargs = {
            "ckpt_path": args.sckite_checkpoint_path,
            "vocab_path": args.sckite_vocab_path,
            "model_py_path": args.sckite_model_path,
            "gears_hidden_size": args.hidden_size,
            "num_bins": args.num_bins,
            "contextual_value_mode": args.contextual_value_mode,
            "contextual_max_genes": None if args.contextual_max_genes <= 0 else args.contextual_max_genes,
            "contextual_gene_selection": args.contextual_gene_selection,
            "contextual_fallback": args.contextual_fallback,
        }

        if args.gene_embedding_mode == "sckite_contextual":
            if args.sckite_model_path is None:
                raise ValueError("--sckite_model_path is required when gene_embedding_mode='sckite_contextual'.")
            adapter_kwargs["model_py_path"] = args.sckite_model_path
            adapter_kwargs["num_bins"] = args.num_bins
            model_kwargs = load_json_arg(args.sckite_model_kwargs_json)
            if model_kwargs is not None:
                adapter_kwargs["model_kwargs"] = model_kwargs

        return adapter_kwargs

    raise ValueError(f"Unknown adapter_name: {args.adapter_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="scKITE-GEARS training script.")

    parser.add_argument("--data_root", type=str, required=True, help="GEARS data root directory.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory containing perturb_processed.h5ad.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save trained GEARS model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")

    parser.add_argument("--split", type=str, default="simulation", help="GEARS split type.")
    parser.add_argument("--train_gene_set_size", type=float, default=0.75, help="Train gene set size for simulation split.")
    parser.add_argument("--combo_seen2_train_frac", type=float, default=0.75, help="Combo seen2 train fraction.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Validation/test batch size.")

    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")

    parser.add_argument("--hidden_size", type=int, default=64, help="GEARS hidden size.")
    parser.add_argument("--num_go_gnn_layers", type=int, default=1, help="Number of GO GNN layers.")
    parser.add_argument("--num_gene_gnn_layers", type=int, default=1, help="Number of co-expression GNN layers.")
    parser.add_argument("--decoder_hidden_size", type=int, default=16, help="Gene-specific decoder hidden size.")
    parser.add_argument("--num_similar_genes_go_graph", type=int, default=20, help="K for GO graph.")
    parser.add_argument("--num_similar_genes_co_express_graph", type=int, default=20, help="K for co-expression graph.")
    parser.add_argument("--coexpress_threshold", type=float, default=0.4, help="Co-expression threshold.")
    parser.add_argument("--uncertainty", type=str2bool, default=False, help="Whether to use uncertainty mode.")
    parser.add_argument("--uncertainty_reg", type=float, default=1.0, help="Uncertainty regularization.")
    parser.add_argument("--direction_lambda", type=float, default=1e-1, help="Direction loss weight.")
    parser.add_argument("--no_perturb", type=str2bool, default=False, help="Predict no perturbation condition.")

    parser.add_argument("--adapter_name", type=str, default="native_gears", choices=["native_gears", "random", "sckite"], help="Adapter name.")
    parser.add_argument("--gene_embedding_mode", type=str, default="native", choices=["native", "sckite_static", "sckite_contextual"], help="Gene embedding mode.")
    parser.add_argument("--pert_embedding_mode", type=str, default="native", choices=["native", "sckite_init"], help="Perturbation embedding mode.")
    parser.add_argument("--freeze_pert_emb", type=str2bool, default=False, help="Freeze pert_emb after scKITE initialization.")

    parser.add_argument("--sckite_checkpoint_path", type=str, default=None, help="Path to the scKITE Stage 2 checkpoint.")
    parser.add_argument("--sckite_vocab_path", type=str, default=None, help="Path to the scKITE gene table.")
    parser.add_argument("--sckite_model_path", type=str, default=None, help="Path to the scKITE Stage 2 model module.")
    parser.add_argument("--sckite_model_kwargs_json", type=str, default=None, help="JSON string or JSON file for ScKITEStage2Model kwargs.")
    parser.add_argument("--num_bins", type=int, default=51, help="Expression bins for the contextual scKITE encoder.")
    parser.add_argument("--project_method", type=str, default="slice", choices=["slice", "mean_pool"], help="Projection method inside adapter.")
    parser.add_argument("--missing_strategy", type=str, default="mean_gene", choices=["mean_gene", "mean_all", "zero"], help="Missing gene embedding strategy.")
    parser.add_argument("--normalize_embedding", type=str2bool, default=False, help="Whether to L2 normalize adapter embeddings.")
    parser.add_argument("--contextual_value_mode", type=str, default="bin")
    parser.add_argument("--contextual_max_genes", type=int, default=0)
    parser.add_argument("--contextual_gene_selection", type=str, default="matched_first")
    parser.add_argument("--contextual_fallback", type=str, default="static")

    parser.add_argument("--wandb", type=str2bool, default=False, help="Whether to use wandb.")
    parser.add_argument("--wandb_project", type=str, default="GEARS", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default="GEARS_adapter_run", help="Wandb run name.")

    parser.add_argument("--save_run_config", type=str2bool, default=True, help="Whether to save run_config.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA is not available. Use CPU instead.", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.save_run_config:
        with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("========== GEARS Adapter Training ==========", flush=True)
    print(f"device: {device}", flush=True)
    print(f"adapter_name: {args.adapter_name}", flush=True)
    print(f"gene_embedding_mode: {args.gene_embedding_mode}", flush=True)
    print(f"pert_embedding_mode: {args.pert_embedding_mode}", flush=True)
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
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    print("Step 4: Build adapter kwargs.", flush=True)
    adapter_kwargs = build_adapter_kwargs(args, device=device)

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
        num_similar_genes_co_express_graph=args.num_similar_genes_co_express_graph,
        coexpress_threshold=args.coexpress_threshold,
        uncertainty=args.uncertainty,
        uncertainty_reg=args.uncertainty_reg,
        direction_lambda=args.direction_lambda,
        no_perturb=args.no_perturb,
        adapter_name=args.adapter_name,
        adapter_kwargs=adapter_kwargs,
        gene_embedding_mode=args.gene_embedding_mode,
        pert_embedding_mode=args.pert_embedding_mode,
        freeze_pert_emb=args.freeze_pert_emb,
    )

    print("Step 6: Train GEARS.", flush=True)
    gears_model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

    print("Step 7: Save model.", flush=True)
    gears_model.save_model(str(save_dir))

    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
