#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from sckite.stage2.tokenizer import GlobalGeneTextTokenizer
from sckite.stage2.data import build_streaming_dataloader
from sckite.stage2.model import (
    ScKITEStage2Model,
    freeze_encoder_backbone,
    load_stage1_encoder_weights,
    masked_mse_loss,
)

try:
    import wandb
except Exception:
    wandb = None


REGULON_TASK_ID = 0
ANNOTATION_TASK_ID = 1
IGNORE_INDEX = -100






def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {} if cfg is None else cfg


def save_json(obj: Any, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_yaml(obj: Dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def make_run_dir(output_root: str, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cleanup_old_epoch_ckpts(run_dir: Path, keep_last_n: int) -> None:
    if int(keep_last_n) <= 0:
        return
    ckpts = sorted(run_dir.glob("epoch_*.pt"))
    for path in ckpts[:-int(keep_last_n)]:
        path.unlink(missing_ok=True)


def safe_div(num: float, den: float) -> float:
    if float(den) <= 0:
        return float("nan")
    return float(num) / float(den)


def format_wandb_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("train_"):
            formatted["train/" + key[len("train_"):]] = value
        elif key.startswith("val_"):
            formatted["val/" + key[len("val_"):]] = value
        else:
            formatted[key] = value
    return formatted






def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp_if_needed() -> Tuple[torch.device, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 多进程训练要求 CUDA 可用。")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, local_rank, use_ddp


def cleanup_ddp() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> ScKITEStage2Model:
    return model.module if isinstance(model, DDP) else model


def reduce_sum_tensor(values: Sequence[float], device: torch.device) -> List[float]:
    tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(x) for x in tensor.cpu().tolist()]


def get_autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()












def _is_gene_id(tokenizer: GlobalGeneTextTokenizer, token_id: int) -> bool:
    try:
        return bool(tokenizer.global_id_is_gene(int(token_id)))
    except Exception:
        return False


def oracle_target_macro_f1_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: GlobalGeneTextTokenizer,
) -> Tuple[float, float]:
    """
    Teacher-forced Oracle Target Macro-F1.

    GT TF queries are already present in each Regulon decoder sequence.
    For every compact Regulon block:
      TF <arrow> T1 T2 ... <regulon_sep>

    target predictions are collected at the GT target-token positions and
    compared as sets within each Regulon block. F1 is macro-averaged over blocks.
    """
    pred_ids = logits.detach().argmax(dim=-1)

    f1_sum = 0.0
    f1_count = 0.0

    for row in range(labels.shape[0]):
        true_row = labels[row]
        pred_row = pred_ids[row]

        state = "tf"
        true_targets: set = set()
        pred_targets: set = set()

        def finalize_block() -> None:
            nonlocal f1_sum, f1_count, true_targets, pred_targets
            if not true_targets:
                true_targets = set()
                pred_targets = set()
                return
            tp = len(true_targets & pred_targets)
            precision = tp / len(pred_targets) if pred_targets else 0.0
            recall = tp / len(true_targets)
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            f1_sum += float(f1)
            f1_count += 1.0
            true_targets = set()
            pred_targets = set()

        for pos in range(true_row.shape[0]):
            true_id = int(true_row[pos].item())
            if true_id == IGNORE_INDEX:
                continue

            if true_id == int(tokenizer.eos_token_id):
                finalize_block()
                break

            if state == "tf":
                if _is_gene_id(tokenizer, true_id):
                    state = "arrow"
                continue

            if state == "arrow":
                if true_id == int(tokenizer.arrow_token_id):
                    state = "targets"
                elif true_id == int(tokenizer.regulon_sep_token_id):
                    state = "tf"
                continue


            if true_id == int(tokenizer.regulon_sep_token_id):
                finalize_block()
                state = "tf"
                continue

            if _is_gene_id(tokenizer, true_id):
                true_targets.add(true_id)
                pred_id = int(pred_row[pos].item())
                if _is_gene_id(tokenizer, pred_id):
                    pred_targets.add(pred_id)

        else:
            finalize_block()

    return f1_sum, f1_count


def global_text_ids_to_text(
    tokenizer: GlobalGeneTextTokenizer,
    ids: Sequence[int],
) -> str:
    bert_tokens: List[str] = []
    prefix = str(tokenizer.text_token_prefix)
    suffix = str(tokenizer.token_suffix)

    for token_id in ids:
        token_id = int(token_id)
        if token_id == IGNORE_INDEX:
            continue
        global_token = tokenizer.global_id_to_token.get(token_id, "")
        if not (global_token.startswith(prefix) and global_token.endswith(suffix)):
            continue
        end = -len(suffix) if len(suffix) > 0 else None
        bert_tokens.append(global_token[len(prefix):end])

    if not bert_tokens:
        return ""
    return tokenizer.text_tokenizer.convert_tokens_to_string(bert_tokens)


def annotation_word_counter(text: str) -> Counter:
    words = re.findall(
        r"[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*",
        str(text).lower(),
    )
    return Counter(words)


def annotation_overlap_f1(
    pred_ids: Sequence[int],
    true_ids: Sequence[int],
    tokenizer: GlobalGeneTextTokenizer,
) -> float:
    pred_text = global_text_ids_to_text(tokenizer, pred_ids)
    true_text = global_text_ids_to_text(tokenizer, true_ids)

    pred_counter = annotation_word_counter(pred_text)
    true_counter = annotation_word_counter(true_text)

    pred_total = sum(pred_counter.values())
    true_total = sum(true_counter.values())

    if pred_total == 0 and true_total == 0:
        return 1.0
    if pred_total == 0 or true_total == 0:
        return 0.0

    overlap = sum((pred_counter & true_counter).values())
    precision = overlap / pred_total
    recall = overlap / true_total
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_best_score(val_metrics: Dict[str, float]) -> float:
    """
    Best checkpoint selection score.

    Higher is better:
      mean(
        Oracle Target Macro-F1,
        Annotation Overlap F1
      )

    Expr MSE is monitored but excluded because its scale is not [0, 1].
    """
    keys = [
        "val_oracle_target_macro_f1",
        "val_annotation_overlap_f1",
    ]
    values = [float(val_metrics.get(k, float("nan"))) for k in keys]
    if not all(math.isfinite(x) for x in values):
        return float("nan")
    return float(sum(values) / len(values))






def get_loss_weights(
    cfg: Dict[str, Any],
) -> Tuple[float, float, float]:
    train_cfg = cfg.get("train", {})
    legacy_decoder = float(train_cfg.get("lambda_decoder", 1.0))

    lambda_expr = float(train_cfg.get("lambda_expr", 1.0))
    lambda_regulon = float(
        train_cfg.get("lambda_regulon", legacy_decoder)
    )
    lambda_annotation = float(
        train_cfg.get("lambda_annotation", legacy_decoder)
    )
    return (
        lambda_expr,
        lambda_regulon,
        lambda_annotation,
    )


def default_decoder_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "name": "regulon",
            "field": "active_regulon_ids",
            "target_type": "multi_regulon_compact",
            "enabled": True,
        },
        {
            "name": "annotation",
            "field": "natural_language_annotation",
            "target_type": "plain_text",
            "enabled": True,
        },
    ]






def build_tokenizer(cfg: Dict[str, Any]) -> GlobalGeneTextTokenizer:
    gv_cfg = cfg.get("global_vocab", {})
    data_cfg = cfg.get("data", {})
    decoder_cfg = cfg.get("decoder", {})

    max_decoder_length = int(
        decoder_cfg.get(
            "max_decoder_length",
            data_cfg.get("max_decoder_length", 1025),
        )
    )

    return GlobalGeneTextTokenizer.from_files(
        text_tokenizer_path=gv_cfg["text_tokenizer_path"],
        global_vocab_path=gv_cfg["global_vocab_path"],
        global_vocab_meta_path=gv_cfg["global_vocab_meta_path"],
        gene_table_path=gv_cfg["gene_table_path"],
        max_length=max_decoder_length,
        use_fast=bool(gv_cfg.get("use_fast", True)),
        do_lower_case=bool(gv_cfg.get("do_lower_case", False)),
    )


def build_loaders(
    cfg: Dict[str, Any],
    tokenizer: GlobalGeneTextTokenizer,
):
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    decoder_cfg = cfg.get("decoder", {})
    decoder_tasks = cfg.get(
        "decoder_tasks",
        default_decoder_tasks(),
    )

    common_kwargs = dict(
        tokenizer=tokenizer,
        max_encoder_length=int(
            data_cfg.get("max_encoder_length", 2049)
        ),
        max_decoder_length=int(
            decoder_cfg.get(
                "max_decoder_length",
                data_cfg.get("max_decoder_length", 1025),
            )
        ),
        mlm_probability=float(
            data_cfg.get("mlm_probability", 0.10)
        ),
        mask_gene_input=bool(
            data_cfg.get("mask_gene_input", False)
        ),
        mask_expr_input=bool(
            data_cfg.get("mask_expr_input", True)
        ),
        num_bins=int(data_cfg.get("num_bins", 51)),
        sampling=bool(data_cfg.get("sampling", False)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        decoder_tasks=decoder_tasks,
        genes_field=str(data_cfg.get("genes_field", "genes")),
        expressions_field=str(
            data_cfg.get("expressions_field", "expressions")
        ),
        loss_weight_field=data_cfg.get(
            "loss_weight_field",
            "loss_weight",
        ),
        keep_first_n_tokens=int(
            data_cfg.get("keep_first_n_tokens", 1)
        ),
        pad_value=float(data_cfg.get("pad_value", -2.0)),
        cls_value=float(data_cfg.get("cls_value", -1.0)),
        mask_value=float(data_cfg.get("mask_value", -3.0)),
        use_loss_weight=bool(
            data_cfg.get("use_loss_weight", False)
        ),
        regulon_target_path=paths_cfg.get(
            "regulon_target_path"
        ),
        active_regulon_field=str(
            data_cfg.get(
                "active_regulon_field",
                "active_regulon_ids",
            )
        ),
        cell_id_field=str(
            data_cfg.get("cell_id_field", "cell_id")
        ),
        n_regulons=int(data_cfg.get("n_regulons", 530)),
        regulon_num_queries=int(
            data_cfg.get("regulon_num_queries", 3)
        ),
        regulon_sampling_mode=str(
            data_cfg.get(
                "regulon_sampling_mode",
                "cyclic_without_replacement",
            )
        ),
        regulon_seed=int(
            data_cfg.get("regulon_seed", 42)
        ),
        validation_regulon_seed=int(
            data_cfg.get("validation_regulon_seed", 2026)
        ),
        cell_specific_targets=bool(
            data_cfg.get("cell_specific_targets", True)
        ),
        expressed_gene_threshold=float(
            data_cfg.get("expressed_gene_threshold", 0.0)
        ),
        dynamic_target_budget=bool(
            data_cfg.get("dynamic_target_budget", True)
        ),
        fixed_eval_encoder_mask=bool(
            data_cfg.get("fixed_eval_encoder_mask", True)
        ),
    )

    batch_size = int(data_cfg.get("batch_size", 16))
    eval_batch_size = int(
        data_cfg.get("eval_batch_size", batch_size)
    )

    train_loader = None
    val_loader = None

    if paths_cfg.get("train_local") is not None:
        train_loader = build_streaming_dataloader(
            local=paths_cfg["train_local"],
            batch_size=batch_size,
            shuffle=True,
            task_sampling_mode=str(
                data_cfg.get("task_sampling_mode", "all")
            ),
            is_training=True,
            **common_kwargs,
        )

    val_path = paths_cfg.get(
        "val_local",
        paths_cfg.get("valid_local", None),
    )
    if val_path is not None:
        val_loader = build_streaming_dataloader(
            local=val_path,
            batch_size=eval_batch_size,
            shuffle=False,
            task_sampling_mode="all",
            is_training=False,
            **common_kwargs,
        )

    return train_loader, val_loader


def build_model(
    cfg: Dict[str, Any],
    tokenizer: GlobalGeneTextTokenizer,
    device: torch.device,
) -> ScKITEStage2Model:
    model_cfg = dict(cfg.get("model", {}))
    data_cfg = cfg.get("data", {})
    decoder_cfg = cfg.get("decoder", {})

    stage1_ckpt_path = model_cfg.pop(
        "stage1_ckpt_path",
        None,
    )
    freeze_encoder = bool(
        model_cfg.pop("freeze_encoder", False)
    )
    freeze_embeddings = bool(
        model_cfg.pop("freeze_embeddings", False)
    )
    freeze_value_head = bool(
        model_cfg.pop("freeze_value_head", False)
    )

    model_cfg.pop("gene_vocab_size", None)
    model_cfg.pop("text_vocab_size", None)
    model_cfg.pop("vocab_size", None)

    model_cfg["global_vocab_size"] = int(
        tokenizer.vocab_size
    )
    model_cfg["pad_token_id"] = int(
        tokenizer.pad_token_id
    )
    model_cfg["mask_value"] = float(
        data_cfg.get("mask_value", -3.0)
    )
    model_cfg["max_decoder_length"] = int(
        decoder_cfg.get(
            "max_decoder_length",
            data_cfg.get("max_decoder_length", 1025),
        )
    )
    model = ScKITEStage2Model(**model_cfg)

    if stage1_ckpt_path:
        load_stage1_encoder_weights(
            model,
            stage1_ckpt_path,
            strict_encoder=False,
            verbose=is_main_process(),
        )

    if freeze_encoder:
        freeze_encoder_backbone(
            model,
            freeze_embeddings=freeze_embeddings,
            freeze_value_head=freeze_value_head,
        )

    model.to(device)
    return model


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = (
            value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
    return out






def decoder_ce_row_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_size = logits.shape[-1]

    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=int(ignore_index),
        reduction="none",
    ).reshape(labels.shape)

    valid = labels.ne(int(ignore_index))
    valid_rows = valid.any(dim=1)
    token_count = valid.sum(dim=1).clamp_min(1)

    row_losses = (
        token_loss * valid.float()
    ).sum(dim=1) / token_count.float()

    return row_losses, valid_rows


def mean_task_decoder_loss(
    logits: Optional[torch.Tensor],
    batch_indices: Optional[torch.Tensor],
    labels: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if (
        logits is None
        or batch_indices is None
        or int(batch_indices.numel()) == 0
    ):
        zero = reference.sum() * 0.0
        return zero, None, None

    task_labels = labels.index_select(
        0,
        batch_indices.long(),
    )
    row_losses, valid_rows = decoder_ce_row_losses(
        logits,
        task_labels,
    )

    if valid_rows.any():
        loss = row_losses[valid_rows].mean()
    else:
        loss = reference.sum() * 0.0

    return loss, row_losses, valid_rows




def compute_losses(
    outputs: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    (
        lambda_expr,
        lambda_regulon,
        lambda_annotation,
    ) = get_loss_weights(cfg)

    reference = outputs["encoder_outputs"]

    if lambda_expr > 0:
        loss_expr = masked_mse_loss(
            outputs["expr_preds"],
            batch["encoder_target_values"],
            batch["expr_masks"],
        )
    else:
        loss_expr = reference.sum() * 0.0

    loss_regulon, _, _ = mean_task_decoder_loss(
        outputs.get("regulon_logits"),
        outputs.get("regulon_batch_indices"),
        batch["decoder_labels"],
        reference,
    )

    loss_annotation, _, _ = mean_task_decoder_loss(
        outputs.get("annotation_logits"),
        outputs.get("annotation_batch_indices"),
        batch["decoder_labels"],
        reference,
    )

    total_loss = (
        lambda_expr * loss_expr
        + lambda_regulon * loss_regulon
        + lambda_annotation * loss_annotation
    )

    return {
        "loss": total_loss,
        "loss_expr": loss_expr,
        "loss_regulon": loss_regulon,
        "loss_annotation": loss_annotation,
    }


def forward_one_batch(
    model: ScKITEStage2Model,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    (
        _,
        lambda_regulon,
        lambda_annotation,
    ) = get_loss_weights(cfg)

    use_decoder = (
        lambda_regulon > 0
        or lambda_annotation > 0
    )

    outputs = model(
        encoder_input_gene_ids=batch[
            "encoder_input_gene_ids"
        ],
        encoder_input_values=batch[
            "encoder_input_values"
        ],
        encoder_key_padding_mask=batch[
            "encoder_key_padding_mask"
        ],
        decoder_input_ids=(
            batch["decoder_input_ids"]
            if use_decoder
            else None
        ),
        decoder_attention_mask=(
            batch["decoder_attention_mask"]
            if use_decoder
            else None
        ),
        decoder_task_ids=(
            batch["decoder_task_ids"]
            if use_decoder
            else None
        ),
        return_encoder_gene_logits=False,
    )

    losses = compute_losses(
        outputs,
        batch,
        cfg,
    )
    return outputs, losses






@torch.no_grad()
def evaluate_validation(
    model: ScKITEStage2Model,
    loader,
    device: torch.device,
    cfg: Dict[str, Any],
    tokenizer: GlobalGeneTextTokenizer,
) -> Dict[str, float]:
    """
    Validation metrics:
      expr_mse
      oracle_target_macro_f1
      annotation_overlap_f1
    """
    train_cfg = cfg.get("train", {})
    use_amp = (
        bool(train_cfg.get("amp", True))
        and device.type == "cuda"
    )

    model.eval()

    expr_sq_sum = 0.0
    expr_count = 0.0

    oracle_target_f1_sum = 0.0
    oracle_target_f1_count = 0.0

    annotation_overlap_sum = 0.0
    annotation_overlap_count = 0.0

    pbar = tqdm(
        loader,
        desc="validation",
        leave=False,
        disable=not is_main_process(),
    )

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        with get_autocast_context(device, use_amp):
            outputs, _ = forward_one_batch(
                model,
                batch,
                cfg,
            )

        expr_mask = batch["expr_masks"].bool()
        if expr_mask.any():
            diff = (
                outputs["expr_preds"].detach().float()[expr_mask]
                - batch["encoder_target_values"].detach().float()[expr_mask]
            )
            expr_sq_sum += float(diff.pow(2).sum().item())
            expr_count += float(diff.numel())

        reg_logits = outputs.get("regulon_logits")
        reg_indices = outputs.get("regulon_batch_indices")

        if (
            reg_logits is not None
            and reg_indices is not None
            and reg_indices.numel() > 0
        ):
            reg_labels = batch["decoder_labels"].index_select(
                0,
                reg_indices.long(),
            )
            f1_sum, f1_count = oracle_target_macro_f1_counts(
                reg_logits,
                reg_labels,
                tokenizer,
            )
            oracle_target_f1_sum += f1_sum
            oracle_target_f1_count += f1_count

        ann_logits = outputs.get("annotation_logits")
        ann_indices = outputs.get("annotation_batch_indices")

        if (
            ann_logits is not None
            and ann_indices is not None
            and ann_indices.numel() > 0
        ):
            ann_labels = batch["decoder_labels"].index_select(
                0,
                ann_indices.long(),
            )
            ann_pred = ann_logits.detach().argmax(dim=-1)

            for row in range(ann_labels.shape[0]):
                valid = ann_labels[row].ne(IGNORE_INDEX)
                if not valid.any():
                    continue

                pred_ids = ann_pred[row][valid].detach().cpu().tolist()
                true_ids = ann_labels[row][valid].detach().cpu().tolist()

                annotation_overlap_sum += annotation_overlap_f1(
                    pred_ids,
                    true_ids,
                    tokenizer,
                )
                annotation_overlap_count += 1.0

    reduced = reduce_sum_tensor(
        [
            expr_sq_sum,
            expr_count,
            oracle_target_f1_sum,
            oracle_target_f1_count,
            annotation_overlap_sum,
            annotation_overlap_count,
        ],
        device=device,
    )

    (
        expr_sq_sum,
        expr_count,
        oracle_target_f1_sum,
        oracle_target_f1_count,
        annotation_overlap_sum,
        annotation_overlap_count,
    ) = reduced

    metrics = {
        "val_expr_mse": float(safe_div(expr_sq_sum, expr_count)),
        "val_oracle_target_macro_f1": float(
            safe_div(oracle_target_f1_sum, oracle_target_f1_count)
        ),
        "val_annotation_overlap_f1": float(
            safe_div(annotation_overlap_sum, annotation_overlap_count)
        ),
    }

    model.train()
    return metrics






def save_checkpoint(
    save_path: Path,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_metric: float,
    metrics: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict":
                unwrap_model(model).state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                None
                if scheduler is None
                else scheduler.state_dict(),
            "scaler_state_dict":
                None
                if scaler is None
                else scaler.state_dict(),
            "best_metric":
                float(best_metric),
            "best_metric_name":
                "decoder_knowledge_score",
            "metrics": metrics,
            "config": cfg,
        },
        save_path,
    )


def load_checkpoint_for_resume(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> Tuple[int, int, float]:
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    unwrap_model(model).load_state_dict(
        ckpt["model_state_dict"]
    )
    optimizer.load_state_dict(
        ckpt["optimizer_state_dict"]
    )

    if (
        scheduler is not None
        and ckpt.get(
            "scheduler_state_dict"
        ) is not None
    ):
        scheduler.load_state_dict(
            ckpt[
                "scheduler_state_dict"
            ]
        )

    if (
        scaler is not None
        and ckpt.get(
            "scaler_state_dict"
        ) is not None
    ):
        scaler.load_state_dict(
            ckpt[
                "scaler_state_dict"
            ]
        )

    start_epoch = int(
        ckpt.get("epoch", -1)
    ) + 1

    global_step = int(
        ckpt.get("global_step", 0)
    )

    best_metric = float(
        ckpt.get(
            "best_metric",
            -float("inf"),
        )
    )

    return (
        start_epoch,
        global_step,
        best_metric,
    )


def load_init_weights(
    model: torch.nn.Module,
    init_from: str,
    device: torch.device,
) -> None:
    ckpt = torch.load(
        init_from,
        map_location=device,
    )

    raw_state_dict = ckpt.get(
        "model_state_dict",
        ckpt.get(
            "state_dict",
            ckpt,
        ),
    )

    model_obj = unwrap_model(model)
    current_state = model_obj.state_dict()

    compatible_state: Dict[
        str,
        torch.Tensor,
    ] = {}

    skipped_shape: Dict[
        str,
        Tuple[
            Tuple[int, ...],
            Tuple[int, ...],
        ],
    ] = {}

    skipped_missing: List[str] = []

    for key, value in (
        raw_state_dict.items()
    ):
        clean_key = (
            key[7:]
            if str(key).startswith(
                "module."
            )
            else key
        )

        if clean_key not in current_state:
            skipped_missing.append(
                clean_key
            )
            continue

        if (
            tuple(value.shape)
            != tuple(
                current_state[
                    clean_key
                ].shape
            )
        ):
            skipped_shape[
                clean_key
            ] = (
                tuple(value.shape),
                tuple(
                    current_state[
                        clean_key
                    ].shape
                ),
            )
            continue

        compatible_state[
            clean_key
        ] = value

    missing, unexpected = (
        model_obj.load_state_dict(
            compatible_state,
            strict=False,
        )
    )

    if is_main_process():
        print(
            "initialized compatible "
            f"model weights from "
            f"{init_from}"
        )
        print(
            "loaded compatible keys: "
            f"{len(compatible_state)}"
        )
        print(
            "missing keys after "
            f"partial load: "
            f"{len(missing)}"
        )
        print(
            "unexpected keys after "
            f"partial load: "
            f"{len(unexpected)}"
        )
        print(
            "skipped missing-in-current "
            f"keys: "
            f"{len(skipped_missing)}"
        )
        print(
            "skipped shape-mismatch "
            f"keys: "
            f"{len(skipped_shape)}"
        )






def make_rolling_buffers(
    window: int,
) -> Dict[str, Deque[float]]:
    window = max(1, int(window))
    return {
        "loss": deque(maxlen=window),
        "expr_loss": deque(maxlen=window),
        "regulon_loss": deque(maxlen=window),
        "annotation_loss": deque(maxlen=window),
    }


def rolling_mean(
    buffer: Deque[float],
) -> float:
    if not buffer:
        return float("nan")
    return float(
        sum(buffer) / len(buffer)
    )


def train_one_epoch(
    model: ScKITEStage2Model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    cfg: Dict[str, Any],
    tokenizer: GlobalGeneTextTokenizer,
    epoch: int,
    global_step: int,
    rolling_buffers: Dict[
        str,
        Deque[float],
    ],
    val_loader=None,
    run_dir: Optional[Path] = None,
    best_metric: float = -float("inf"),
    history: Optional[
        List[Dict[str, Any]]
    ] = None,
) -> Tuple[
    int,
    Dict[str, float],
    float,
]:
    train_cfg = cfg.get(
        "train",
        {},
    )

    use_amp = (
        bool(train_cfg.get("amp", True))
        and device.type == "cuda"
    )

    grad_clip = float(
        train_cfg.get(
            "grad_clip",
            1.0,
        )
    )

    log_every = int(
        train_cfg.get(
            "log_every_n_steps",
            50,
        )
    )

    wandb_log_every = int(
        train_cfg.get(
            "wandb_log_every_n_steps",
            1,
        )
    )

    eval_every_n_steps = int(
        train_cfg.get(
            "eval_every_n_steps",
            0,
        )
    )

    save_step_ckpt = bool(
        train_cfg.get(
            "save_step_ckpt",
            False,
        )
    )


    collator = getattr(
        loader,
        "collate_fn",
        None,
    )
    if (
        collator is not None
        and hasattr(
            collator,
            "set_epoch",
        )
    ):
        collator.set_epoch(epoch)

    running = {
        "loss_sum": 0.0,
        "loss_expr_sum": 0.0,
        "loss_regulon_sum": 0.0,
        "loss_annotation_sum": 0.0,
        "n_steps": 0.0,
    }

    amp_overflow_skips = 0

    model.train()

    pbar = tqdm(
        loader,
        desc=f"train epoch {epoch}",
        leave=True,
        disable=not is_main_process(),
    )

    for batch in pbar:
        batch = move_batch_to_device(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with get_autocast_context(
            device,
            use_amp,
        ):
            _, loss_pack = (
                forward_one_batch(
                    model,
                    batch,
                    cfg,
                )
            )
            loss = loss_pack["loss"]

        optimizer_was_run = True

        if (
            scaler is not None
            and use_amp
        ):
            scaler.scale(
                loss
            ).backward()

            if grad_clip > 0:
                scaler.unscale_(
                    optimizer
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            scale_before = float(
                scaler.get_scale()
            )

            scaler.step(
                optimizer
            )
            scaler.update()

            scale_after = float(
                scaler.get_scale()
            )


            optimizer_was_run = (
                scale_after
                >= scale_before
            )

            if not optimizer_was_run:
                amp_overflow_skips += 1

                if is_main_process():
                    print(
                        "[AMP overflow] "
                        "optimizer step skipped: "
                        f"scale "
                        f"{scale_before:g} "
                        f"-> "
                        f"{scale_after:g}; "
                        f"skipped_updates="
                        f"{amp_overflow_skips}"
                    )

        else:
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()
            optimizer_was_run = True




        if not optimizer_was_run:
            continue

        if scheduler is not None:
            scheduler.step()

        global_step += 1





        reduced_step = reduce_sum_tensor(
            [
                float(
                    loss_pack[
                        "loss"
                    ].detach().item()
                ),
                float(
                    loss_pack[
                        "loss_expr"
                    ].detach().item()
                ),
                float(
                    loss_pack[
                        "loss_regulon"
                    ].detach().item()
                ),
                float(
                    loss_pack[
                        "loss_annotation"
                    ].detach().item()
                ),
            ],
            device=device,
        )

        world = float(
            get_world_size()
        )
        step_values = [
            x / world
            for x in reduced_step
        ]

        (
            step_loss,
            step_expr,
            step_regulon,
            step_annotation,
        ) = step_values

        rolling_buffers[
            "loss"
        ].append(step_loss)

        rolling_buffers[
            "expr_loss"
        ].append(step_expr)

        rolling_buffers[
            "regulon_loss"
        ].append(step_regulon)

        rolling_buffers[
            "annotation_loss"
        ].append(step_annotation)

        running[
            "loss_sum"
        ] += step_loss

        running[
            "loss_expr_sum"
        ] += step_expr

        running[
            "loss_regulon_sum"
        ] += step_regulon

        running[
            "loss_annotation_sum"
        ] += step_annotation

        running[
            "n_steps"
        ] += 1.0

        rolling_metrics = {
            "train_loss":
                rolling_mean(
                    rolling_buffers[
                        "loss"
                    ]
                ),
            "train_expr_loss":
                rolling_mean(
                    rolling_buffers[
                        "expr_loss"
                    ]
                ),
            "train_regulon_loss":
                rolling_mean(
                    rolling_buffers[
                        "regulon_loss"
                    ]
                ),
            "train_annotation_loss":
                rolling_mean(
                    rolling_buffers[
                        "annotation_loss"
                    ]
                ),
            "encoder_learning_rate":
                float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
            "decoder_learning_rate":
                float(
                    optimizer.param_groups[
                        1
                    ]["lr"]
                ),
        }


        if (
            wandb_log_every > 0
            and global_step
            % wandb_log_every
            == 0
            and is_main_process()
            and wandb is not None
            and wandb.run is not None
        ):
            wandb.log(
                format_wandb_metrics(
                    {
                        **rolling_metrics,
                        "epoch":
                            int(epoch),
                        "global_step":
                            int(global_step),
                    }
                ),
                step=global_step,
            )


        if (
            log_every > 0
            and global_step
            % log_every
            == 0
            and is_main_process()
        ):
            pbar.set_postfix(
                {
                    "loss":
                        f"{rolling_metrics['train_loss']:.4f}",
                    "expr":
                        f"{rolling_metrics['train_expr_loss']:.4f}",
                    "reg":
                        f"{rolling_metrics['train_regulon_loss']:.4f}",
                    "ann":
                        f"{rolling_metrics['train_annotation_loss']:.4f}",
                    "enc_lr":
                        f"{rolling_metrics['encoder_learning_rate']:.2e}",
                    "dec_lr":
                        f"{rolling_metrics['decoder_learning_rate']:.2e}",
                }
            )




        if (
            val_loader is not None
            and eval_every_n_steps > 0
            and global_step
            % eval_every_n_steps
            == 0
        ):
            val_metrics = evaluate_validation(
                model,
                val_loader,
                device,
                cfg,
                tokenizer
            )

            current_score = (
                compute_best_score(
                    val_metrics
                )
            )

            should_save_best = (
                math.isfinite(
                    current_score
                )
                and current_score
                > best_metric
            )

            if should_save_best:
                best_metric = (
                    current_score
                )

            if is_main_process():
                step_metrics = {
                    "epoch":
                        int(epoch),
                    "global_step":
                        int(global_step),
                    **val_metrics,
                }

                print(
                    json.dumps(
                        {
                            **step_metrics,
                            "selection_score":
                                current_score,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                if history is not None:
                    history.append(
                        {
                            **step_metrics,
                            "selection_score":
                                current_score,
                        }
                    )

                    if run_dir is not None:
                        save_json(
                            {
                                "history":
                                    history
                            },
                            run_dir
                            / "metrics_history.json",
                        )

                if (
                    wandb is not None
                    and wandb.run
                    is not None
                ):

                    wandb.log(
                        format_wandb_metrics(
                            step_metrics
                        ),
                        step=global_step,
                    )

                if (
                    should_save_best
                    and run_dir is not None
                ):
                    save_checkpoint(
                        run_dir
                        / "best.pt",
                        epoch,
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_metric,
                        {
                            **step_metrics,
                            "selection_score":
                                current_score,
                        },
                        cfg,
                    )

                    print(
                        "saved best checkpoint "
                        f"(score={current_score:.6f}) "
                        f"to "
                        f"{run_dir / 'best.pt'}"
                    )

                if (
                    save_step_ckpt
                    and run_dir is not None
                ):
                    save_checkpoint(
                        run_dir
                        / f"step_{global_step}.pt",
                        epoch,
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_metric,
                        {
                            **step_metrics,
                            "selection_score":
                                current_score,
                        },
                        cfg,
                    )

            if is_dist_avail_and_initialized():
                dist.barrier()

            model.train()

    n = max(
        running["n_steps"],
        1.0,
    )

    summary = {
        "train_loss":
            running[
                "loss_sum"
            ] / n,
        "train_expr_loss":
            running[
                "loss_expr_sum"
            ] / n,
        "train_regulon_loss":
            running[
                "loss_regulon_sum"
            ] / n,
        "train_annotation_loss":
            running[
                "loss_annotation_sum"
            ] / n,
        "encoder_learning_rate":
            float(
                optimizer.param_groups[
                    0
                ]["lr"]
            ),
        "decoder_learning_rate":
            float(
                optimizer.param_groups[
                    1
                ]["lr"]
            ),
    }

    return (
        global_step,
        summary,
        best_metric,
    )






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train scKITE Stage2 without Regulon Activity Head: "
            "Shared Encoder + Multi-Regulon Decoder + Annotation Decoder. "
            "Only training and validation splits are used."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径。",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "同一新版 Stage2 结构 checkpoint，"
            "用于完整断点续训。"
        ),
    )

    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help=(
            "按同名且同 shape 参数加载模型权重，"
            "不恢复 optimizer。"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(
        args.config
    )

    train_cfg = cfg.get(
        "train",
        {},
    )

    paths_cfg = cfg.get(
        "paths",
        {},
    )

    device, local_rank, use_ddp = (
        setup_ddp_if_needed()
    )

    set_seed(
        int(
            train_cfg.get(
                "seed",
                42,
            )
        )
        + get_rank()
    )

    tokenizer = build_tokenizer(
        cfg
    )

    if is_main_process():
        print(tokenizer)

        (
            lambda_expr,
            lambda_regulon,
            lambda_annotation,
        ) = get_loss_weights(cfg)

        print(
            "[loss weights] "
            f"lambda_expr="
            f"{lambda_expr}, "
            f"lambda_regulon="
            f"{lambda_regulon}, "
            f"lambda_annotation="
            f"{lambda_annotation}"
        )

        print(
            "[best checkpoint] "
            "maximize mean("
            "val_oracle_target_macro_f1, "
            "val_annotation_overlap_f1)"
        )

    (
        train_loader,
        val_loader,
    ) = build_loaders(
        cfg,
        tokenizer,
    )

    if train_loader is None:
        raise ValueError(
            "必须在 paths.train_local "
            "中提供训练 MDS 路径。"
        )

    if is_main_process():
        run_dir = make_run_dir(
            paths_cfg.get(
                "output_root",
                "./runs",
            ),
            paths_cfg.get(
                "run_name",
                "stage2_dual_decoder",
            ),
        )

        shutil.copy2(
            args.config,
            run_dir
            / "config.original.yaml",
        )

        save_yaml(
            cfg,
            run_dir
            / "config.resolved.yaml",
        )
    else:
        run_dir = None

    if is_dist_avail_and_initialized():
        obj_list = [
            str(run_dir)
            if is_main_process()
            else ""
        ]

        dist.broadcast_object_list(
            obj_list,
            src=0,
        )

        run_dir = Path(
            obj_list[0]
        )

    model = build_model(
        cfg,
        tokenizer,
        device,
    )

    if args.init_from is not None:
        load_init_weights(
            model,
            args.init_from,
            device,
        )

    if use_ddp:
        model = DDP(
            model,
            device_ids=[
                local_rank
            ],
            output_device=
                local_rank,
            find_unused_parameters=bool(
                train_cfg.get(
                    "find_unused_parameters",
                    True,
                )
            ),
        )

        if is_main_process():
            print(
                "Using DDP: "
                f"world_size="
                f"{get_world_size()}, "
                f"local_rank="
                f"{local_rank}"
            )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    n_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if is_main_process():
        print(
            f"device={device}"
        )
        print(
            f"n_params="
            f"{n_params:,}, "
            f"n_trainable="
            f"{n_trainable:,}"
        )

        if use_ddp:
            print(
                "YAML data.batch_size "
                "表示每张 GPU 的 raw-cell "
                "DataLoader batch size；"
                "task_sampling_mode=all 时，"
                "实际模型 batch 可因 Decoder task 展开而增大。"
            )










    encoder_lr = float(
        train_cfg.get(
            "encoder_lr",
            1.0e-5,
        )
    )
    decoder_lr = float(
        train_cfg.get(
            "decoder_lr",
            1.0e-4,
        )
    )
    weight_decay = float(
        train_cfg.get(
            "weight_decay",
            0.01,
        )
    )

    model_obj = unwrap_model(model)

    decoder_prefixes = (
        "regulon_decoder_position_embedding.",
        "regulon_decoder_token_norm.",
        "regulon_decoder.",
        "regulon_lm_head.",
        "annotation_decoder_position_embedding.",
        "annotation_decoder_token_norm.",
        "annotation_decoder.",
        "annotation_lm_head.",
    )

    encoder_params = []
    decoder_params = []

    for name, param in model_obj.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith(decoder_prefixes):
            decoder_params.append(param)
        else:
            encoder_params.append(param)

    if not encoder_params:
        raise RuntimeError(
            "Encoder parameter group is empty."
        )

    if not decoder_params:
        raise RuntimeError(
            "Decoder parameter group is empty. "
            "Please check current model parameter names."
        )

    optimizer = AdamW(
        [
            {
                "params": encoder_params,
                "lr": encoder_lr,
            },
            {
                "params": decoder_params,
                "lr": decoder_lr,
            },
        ],
        weight_decay=weight_decay,
    )

    if is_main_process():
        n_encoder_group = sum(
            p.numel()
            for p in encoder_params
        )
        n_decoder_group = sum(
            p.numel()
            for p in decoder_params
        )

        print(
            "[learning rates] "
            f"encoder_lr={encoder_lr:.2e}, "
            f"decoder_lr={decoder_lr:.2e}"
        )
        print(
            "[optimizer groups] "
            f"encoder_params={n_encoder_group:,}, "
            f"decoder_params={n_decoder_group:,}"
        )

    epochs = int(
        train_cfg.get(
            "epochs",
            10,
        )
    )

    if hasattr(
        train_loader,
        "__len__",
    ):
        steps_per_epoch = len(
            train_loader
        )
    else:
        steps_per_epoch = int(
            train_cfg.get(
                "steps_per_epoch",
                1000,
            )
        )

    total_steps = max(
        1,
        epochs
        * int(
            steps_per_epoch
        ),
    )

    scheduler = (
        CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(
                train_cfg.get(
                    "min_lr",
                    0.0,
                )
            ),
        )
    )

    use_amp = (
        bool(
            train_cfg.get(
                "amp",
                True,
            )
        )
        and device.type
        == "cuda"
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
        if device.type
        == "cuda"
        else None
    )

    start_epoch = 0
    global_step = 0


    best_metric = -float("inf")

    if args.resume is not None:
        (
            start_epoch,
            global_step,
            best_metric,
        ) = load_checkpoint_for_resume(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

        if is_main_process():
            print(
                f"resumed from "
                f"{args.resume}: "
                f"start_epoch="
                f"{start_epoch}, "
                f"global_step="
                f"{global_step}, "
                f"best_score="
                f"{best_metric}"
            )

    wandb_cfg = cfg.get(
        "wandb",
        {},
    )

    if (
        bool(
            wandb_cfg.get(
                "use_wandb",
                False,
            )
        )
        and is_main_process()
    ):
        if wandb is None:
            print(
                "wandb 未安装，"
                "跳过 wandb 记录。"
            )
        else:
            wandb.init(
                project=wandb_cfg.get(
                    "project",
                    "scKITE_stage2",
                ),
                name=wandb_cfg.get(
                    "name",
                    run_dir.name,
                ),
                config=cfg,
                dir=wandb_cfg.get(
                    "dir",
                    str(run_dir),
                ),
            )

    history: List[
        Dict[str, Any]
    ] = []

    keep_last_n = int(
        train_cfg.get(
            "keep_last_n_ckpts",
            3,
        )
    )

    save_every = int(
        train_cfg.get(
            "save_every_n_epochs",
            1,
        )
    )

    rolling_window = int(
        train_cfg.get(
            "wandb_loss_window",
            100,
        )
    )

    rolling_buffers = (
        make_rolling_buffers(
            rolling_window
        )
    )

    try:
        for epoch in range(
            start_epoch,
            epochs,
        ):
            (
                global_step,
                train_metrics,
                best_metric,
            ) = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                cfg=cfg,
                tokenizer=tokenizer,
                epoch=epoch,
                global_step=global_step,
                rolling_buffers=(
                    rolling_buffers
                ),
                val_loader=val_loader,
                run_dir=run_dir,
                best_metric=best_metric,
                history=(
                    history
                    if is_main_process()
                    else None
                ),
            )

            epoch_record: Dict[
                str,
                Any,
            ] = {
                "epoch":
                    int(epoch),
                "global_step":
                    int(global_step),
                "epoch_train_summary":
                    train_metrics,
            }




            if val_loader is not None:
                val_metrics = (
                    evaluate_validation(
                        model,
                        val_loader,
                        device,
                        cfg,
                        tokenizer
            )
                )

                epoch_record.update(
                    val_metrics
                )

                current_score = (
                    compute_best_score(
                        val_metrics
                    )
                )

                epoch_record[
                    "selection_score"
                ] = current_score

                should_save_best = (
                    math.isfinite(
                        current_score
                    )
                    and current_score
                    > best_metric
                )

                if should_save_best:
                    best_metric = (
                        current_score
                    )

            else:
                should_save_best = False
                current_score = float(
                    "nan"
                )

            if is_main_process():
                print(
                    json.dumps(
                        epoch_record,
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                history.append(
                    epoch_record
                )

                save_json(
                    {
                        "history":
                            history
                    },
                    run_dir
                    / "metrics_history.json",
                )


                if (
                    val_loader is not None
                    and wandb is not None
                    and wandb.run
                    is not None
                ):
                    wandb.log(
                        format_wandb_metrics(
                            val_metrics
                        ),
                        step=global_step,
                    )

                if should_save_best:
                    save_checkpoint(
                        run_dir
                        / "best.pt",
                        epoch,
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_metric,
                        epoch_record,
                        cfg,
                    )

                    print(
                        "saved best checkpoint "
                        f"(score="
                        f"{current_score:.6f}) "
                        f"to "
                        f"{run_dir / 'best.pt'}"
                    )

                if (
                    save_every > 0
                    and (epoch + 1)
                    % save_every
                    == 0
                ):
                    ckpt_path = (
                        run_dir
                        / f"epoch_{epoch:04d}.pt"
                    )

                    save_checkpoint(
                        ckpt_path,
                        epoch,
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_metric,
                        epoch_record,
                        cfg,
                    )

                    cleanup_old_epoch_ckpts(
                        run_dir,
                        keep_last_n,
                    )

            if is_dist_avail_and_initialized():
                dist.barrier()

    finally:
        if (
            wandb is not None
            and wandb.run is not None
            and is_main_process()
        ):
            wandb.finish()

        cleanup_ddp()


if __name__ == "__main__":
    main()
