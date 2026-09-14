#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from adapters.scfoundation_adapter import ScFoundationAdapter
from gears import GEARS, PertData
from gears.utils import get_genes_from_perts


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
        description=(
            "Train the existing GEARS perturbation workflow with a frozen "
            "scFoundation gene-embedding adapter."
        )
    )

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--split", default="simulation")
    parser.add_argument("--train_gene_set_size", type=float, default=0.75)
    parser.add_argument("--combo_seen2_train_frac", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=2)

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

    parser.add_argument("--scfoundation_source_dir", required=True)
    parser.add_argument("--scfoundation_ckpt", required=True)
    parser.add_argument("--canonical_gene_path", required=True)
    parser.add_argument("--hgnc_mapping_path", required=True)
    parser.add_argument(
        "--hgnc_sha256",
        default=ScFoundationAdapter.DEFAULT_HGNC_SHA256,
        help="Expected HGNC TSV SHA256; pass an empty value to disable the check.",
    )
    parser.add_argument("--checkpoint_key", choices=["gene"], default="gene")
    parser.add_argument(
        "--pre_normalized",
        choices=["T"],
        default="T",
        help=(
            "T means adata.X is already normalized+log1p. The unchanged GEARS "
            "cell graphs do not carry the raw-count total required by A mode."
        ),
    )
    parser.add_argument("--target_high_resolution", type=float, default=4.0)
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
        "--contextual_fallback", choices=["static", "zero"], default="static"
    )
    parser.add_argument("--contextual_encoder_batch_size", type=int, default=1)
    parser.add_argument(
        "--missing_strategy", choices=["mean_gene", "zero"], default="mean_gene"
    )
    parser.add_argument(
        "--scfoundation_precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
    )
    parser.add_argument("--verify_loaded_tensors", type=str2bool, default=True)
    parser.add_argument("--require_all_perturbations", type=str2bool, default=True)

    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", default="GEARS_scFoundation")
    parser.add_argument("--wandb_run_name", default="GEARS_scFoundation_run")
    parser.add_argument("--save_run_config", type=str2bool, default=True)
    return parser.parse_args()


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.hidden_size != 512:
        raise ValueError(
            "This scFoundation checkpoint emits 512-dimensional decoder "
            "embeddings, so --hidden_size must be 512."
        )

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA is unavailable; falling back to CPU.", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_run_config:
        run_config = vars(args).copy()
        run_config["resolved_device"] = device
        write_json(save_dir / "run_config.json", run_config)

    print("======= GEARS + frozen scFoundation training =======", flush=True)
    print(f"device: {device}", flush=True)
    print(f"gene_embedding_mode: {args.gene_embedding_mode}", flush=True)
    print(f"pert_embedding_mode: {args.pert_embedding_mode}", flush=True)
    print(f"pre_normalized: {args.pre_normalized}", flush=True)
    print(f"target_high_resolution: t{args.target_high_resolution:g}", flush=True)
    print(f"hidden_size: {args.hidden_size}", flush=True)
    print(f"save_dir: {save_dir}", flush=True)

    print("Step 1: Load PertData.", flush=True)
    pert_data = PertData(data_path=args.data_root)
    pert_data.load(data_path=args.dataset_dir)
    required_perturbation_genes = [
        str(gene)
        for gene in get_genes_from_perts(
            pert_data.adata.obs["condition"].astype(str).values
        )
    ]
    print(
        "Dataset perturbation genes retained by GEARS: "
        f"{len(required_perturbation_genes)}",
        flush=True,
    )

    print("Step 2: Prepare the unchanged GEARS split.", flush=True)
    pert_data.prepare_split(
        split=args.split,
        seed=args.seed,
        train_gene_set_size=args.train_gene_set_size,
        combo_seen2_train_frac=args.combo_seen2_train_frac,
    )

    print("Step 3: Build the unchanged GEARS dataloaders.", flush=True)
    pert_data.get_dataloader(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )

    print("Step 4: Build the frozen scFoundation adapter.", flush=True)
    adapter_kwargs = {
        "source_dir": args.scfoundation_source_dir,
        "ckpt_path": args.scfoundation_ckpt,
        "canonical_gene_path": args.canonical_gene_path,
        "hgnc_mapping_path": args.hgnc_mapping_path,
        "device": device,
        "gears_hidden_size": args.hidden_size,
        "checkpoint_key": args.checkpoint_key,
        "pre_normalized": args.pre_normalized,
        "target_high_resolution": args.target_high_resolution,
        "contextual_fallback": args.contextual_fallback,
        "contextual_encoder_batch_size": args.contextual_encoder_batch_size,
        "missing_strategy": args.missing_strategy,
        "precision": args.scfoundation_precision,
        "mapping_report_dir": str(save_dir),
        "expected_hgnc_sha256": args.hgnc_sha256 or None,
        "verify_loaded_tensors": args.verify_loaded_tensors,
        "require_all_perturbations": args.require_all_perturbations,
        "required_perturbation_genes": required_perturbation_genes,
        "seed": args.seed,
    }
    adapter = ScFoundationAdapter(**adapter_kwargs)

    print("Step 5: Initialize the unchanged GEARS model.", flush=True)
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
        adapter_name="scfoundation",
        adapter_kwargs=adapter_kwargs,
        gene_embedding_mode=args.gene_embedding_mode,
        pert_embedding_mode=args.pert_embedding_mode,
        freeze_pert_emb=args.freeze_pert_emb,
        num_bins=100,
        contextual_value_mode="scfoundation_autobin_T",
        contextual_max_genes=None,
        contextual_gene_selection="all",
        contextual_fallback=args.contextual_fallback,
    )
    write_json(
        save_dir / "scfoundation_adapter_config.json", adapter.get_config()
    )

    print("Step 6: Train GEARS.", flush=True)
    gears_model.train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Step 7: Save GEARS and the final adapter metadata.", flush=True)
    gears_model.save_model(str(save_dir))
    write_json(
        save_dir / "scfoundation_adapter_config.json", adapter.get_config()
    )
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
