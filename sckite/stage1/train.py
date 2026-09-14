#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from sckite.stage1.tokenizer import GlobalGeneTextTokenizer
from sckite.stage1.data import build_streaming_dataloader
from sckite.stage1.model import (
    ScKITEStage1Model,
    freeze_encoder_backbone,
    load_stage1_encoder_weights,
    masked_mse_loss,
)

try:
    import wandb
except Exception:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_config(
    config_path: str,
) -> Dict[str, Any]:
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        cfg = yaml.safe_load(file)
    return {} if cfg is None else cfg


def save_json(
    obj: Any,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            obj,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_yaml(
    obj: Dict[str, Any],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            obj,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def make_run_dir(
    output_root: str,
    run_name: str,
) -> Path:
    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = (
        Path(output_root)
        / f"{run_name}_{timestamp}"
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return run_dir


def cleanup_old_epoch_ckpts(
    run_dir: Path,
    keep_last_n: int,
) -> None:
    if int(keep_last_n) <= 0:
        return
    checkpoints = sorted(
        run_dir.glob("epoch_*.pt")
    )
    for path in checkpoints[
        :-int(keep_last_n)
    ]:
        path.unlink(missing_ok=True)


def format_wandb_metrics(
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("train_"):
            output[
                "train/"
                + key[len("train_"):]
            ] = value
        elif key.startswith("val_"):
            output[
                "val/"
                + key[len("val_"):]
            ] = value
        else:
            output[key] = value
    return output


def is_dist_avail_and_initialized() -> bool:
    return (
        dist.is_available()
        and dist.is_initialized()
    )


def get_rank() -> int:
    return (
        dist.get_rank()
        if is_dist_avail_and_initialized()
        else 0
    )


def get_world_size() -> int:
    return (
        dist.get_world_size()
        if is_dist_avail_and_initialized()
        else 1
    )


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp_if_needed() -> Tuple[
    torch.device,
    int,
    bool,
]:
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )
    use_ddp = world_size > 1
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DDP 多进程训练要求 CUDA 可用。"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        device = torch.device(
            "cuda",
            local_rank,
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return device, local_rank, use_ddp


def cleanup_ddp() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(
    model: torch.nn.Module,
) -> ScKITEStage1Model:
    return (
        model.module
        if isinstance(model, DDP)
        else model
    )


def reduce_sum_tensor(
    values: Sequence[float],
    device: torch.device,
) -> List[float]:
    tensor = torch.tensor(
        list(values),
        dtype=torch.float64,
        device=device,
    )
    if is_dist_avail_and_initialized():
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )
    return [
        float(x)
        for x in tensor.cpu().tolist()
    ]


def get_autocast_context(
    device: torch.device,
    enabled: bool,
):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


def build_tokenizer(
    cfg: Dict[str, Any],
) -> GlobalGeneTextTokenizer:
    gv_cfg = cfg.get(
        "global_vocab",
        {},
    )
    return (
        GlobalGeneTextTokenizer
        .from_files(
            text_tokenizer_path=
                gv_cfg[
                    "text_tokenizer_path"
                ],
            global_vocab_path=
                gv_cfg[
                    "global_vocab_path"
                ],
            global_vocab_meta_path=
                gv_cfg[
                    "global_vocab_meta_path"
                ],
            gene_table_path=
                gv_cfg[
                    "gene_table_path"
                ],
            max_length=128,
            use_fast=bool(
                gv_cfg.get(
                    "use_fast",
                    True,
                )
            ),
            do_lower_case=bool(
                gv_cfg.get(
                    "do_lower_case",
                    False,
                )
            ),
        )
    )


def build_loaders(
    cfg: Dict[str, Any],
    tokenizer: GlobalGeneTextTokenizer,
):
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})

    common_kwargs = dict(
        tokenizer=tokenizer,
        max_encoder_length=int(
            data_cfg.get(
                "max_encoder_length",
                2049,
            )
        ),
        mlm_probability=float(
            data_cfg.get(
                "mlm_probability",
                0.10,
            )
        ),
        mask_gene_input=bool(
            data_cfg.get(
                "mask_gene_input",
                False,
            )
        ),
        mask_expr_input=bool(
            data_cfg.get(
                "mask_expr_input",
                True,
            )
        ),
        num_bins=int(
            data_cfg.get(
                "num_bins",
                51,
            )
        ),
        sampling=bool(
            data_cfg.get(
                "sampling",
                False,
            )
        ),
        num_workers=int(
            data_cfg.get(
                "num_workers",
                0,
            )
        ),
        pin_memory=bool(
            data_cfg.get(
                "pin_memory",
                False,
            )
        ),
        drop_last=bool(
            data_cfg.get(
                "drop_last",
                False,
            )
        ),
        genes_field=str(
            data_cfg.get(
                "genes_field",
                "genes",
            )
        ),
        expressions_field=str(
            data_cfg.get(
                "expressions_field",
                "expressions",
            )
        ),
        cell_id_field=str(
            data_cfg.get(
                "cell_id_field",
                "cell_id",
            )
        ),
        keep_first_n_tokens=int(
            data_cfg.get(
                "keep_first_n_tokens",
                1,
            )
        ),
        pad_value=float(
            data_cfg.get(
                "pad_value",
                -2.0,
            )
        ),
        cls_value=float(
            data_cfg.get(
                "cls_value",
                -1.0,
            )
        ),
        mask_value=float(
            data_cfg.get(
                "mask_value",
                -3.0,
            )
        ),
        fixed_eval_encoder_mask=bool(
            data_cfg.get(
                "fixed_eval_encoder_mask",
                True,
            )
        ),
    )

    batch_size = int(
        data_cfg.get(
            "batch_size",
            16,
        )
    )
    eval_batch_size = int(
        data_cfg.get(
            "eval_batch_size",
            batch_size,
        )
    )

    train_loader = (
        build_streaming_dataloader(
            local=paths_cfg["train_local"],
            batch_size=batch_size,
            shuffle=True,
            is_training=True,
            **common_kwargs,
        )
    )

    val_loader = (
        build_streaming_dataloader(
            local=paths_cfg["val_local"],
            batch_size=eval_batch_size,
            shuffle=False,
            is_training=False,
            **common_kwargs,
        )
    )

    return train_loader, val_loader


def build_model(
    cfg: Dict[str, Any],
    tokenizer: GlobalGeneTextTokenizer,
    device: torch.device,
) -> ScKITEStage1Model:
    model_cfg = dict(
        cfg.get(
            "model",
            {},
        )
    )
    data_cfg = cfg.get("data", {})

    stage1_ckpt_path = model_cfg.pop(
        "stage1_ckpt_path",
        None,
    )
    freeze_encoder = bool(
        model_cfg.pop(
            "freeze_encoder",
            False,
        )
    )
    freeze_embeddings = bool(
        model_cfg.pop(
            "freeze_embeddings",
            False,
        )
    )
    freeze_value_head = bool(
        model_cfg.pop(
            "freeze_value_head",
            False,
        )
    )

    model_cfg.pop(
        "gene_vocab_size",
        None,
    )
    model_cfg.pop(
        "text_vocab_size",
        None,
    )
    model_cfg.pop(
        "vocab_size",
        None,
    )

    model_cfg["global_vocab_size"] = int(
        tokenizer.vocab_size
    )
    model_cfg["pad_token_id"] = int(
        tokenizer.pad_token_id
    )
    model_cfg["mask_value"] = float(
        data_cfg.get(
            "mask_value",
            -3.0,
        )
    )

    model = ScKITEStage1Model(
        **model_cfg
    )

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
            freeze_embeddings=
                freeze_embeddings,
            freeze_value_head=
                freeze_value_head,
        )

    model.to(device)
    return model


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        output[key] = (
            value.to(
                device,
                non_blocking=True,
            )
            if torch.is_tensor(value)
            else value
        )
    return output


def forward_one_batch(
    model: ScKITEStage1Model,
    batch: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    torch.Tensor,
]:
    outputs = model(
        encoder_input_gene_ids=
            batch[
                "encoder_input_gene_ids"
            ],
        encoder_input_values=
            batch[
                "encoder_input_values"
            ],
        encoder_key_padding_mask=
            batch[
                "encoder_key_padding_mask"
            ],
        return_last_attn=False,
    )

    loss = masked_mse_loss(
        outputs["expr_preds"],
        batch[
            "encoder_target_values"
        ],
        batch["expr_masks"],
    )
    return outputs, loss


@torch.no_grad()
def evaluate_validation(
    model: ScKITEStage1Model,
    loader,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    train_cfg = cfg.get("train", {})
    use_amp = (
        bool(
            train_cfg.get(
                "amp",
                True,
            )
        )
        and device.type == "cuda"
    )

    model.eval()
    squared_error_sum = 0.0
    value_count = 0.0

    progress = tqdm(
        loader,
        desc="validation",
        leave=False,
        disable=not is_main_process(),
    )

    for batch in progress:
        batch = move_batch_to_device(
            batch,
            device,
        )

        with get_autocast_context(
            device,
            use_amp,
        ):
            outputs, _ = forward_one_batch(
                model,
                batch,
            )

        mask = batch["expr_masks"].bool()
        if mask.any():
            difference = (
                outputs["expr_preds"]
                .detach()
                .float()[mask]
                - batch[
                    "encoder_target_values"
                ]
                .detach()
                .float()[mask]
            )
            squared_error_sum += float(
                difference.pow(2).sum().item()
            )
            value_count += float(
                difference.numel()
            )

    squared_error_sum, value_count = (
        reduce_sum_tensor(
            [
                squared_error_sum,
                value_count,
            ],
            device=device,
        )
    )

    val_expr_mse = (
        squared_error_sum
        / value_count
        if value_count > 0
        else float("nan")
    )

    model.train()
    return {
        "val_expr_mse":
            float(val_expr_mse)
    }


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
                unwrap_model(model)
                .state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                (
                    None
                    if scheduler is None
                    else scheduler.state_dict()
                ),
            "scaler_state_dict":
                (
                    None
                    if scaler is None
                    else scaler.state_dict()
                ),
            "best_metric":
                float(best_metric),
            "best_metric_name":
                "val_expr_mse",
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
    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
    )

    unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    if (
        scheduler is not None
        and checkpoint.get(
            "scheduler_state_dict"
        ) is not None
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (
        scaler is not None
        and checkpoint.get(
            "scaler_state_dict"
        ) is not None
    ):
        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    return (
        int(
            checkpoint.get(
                "epoch",
                -1,
            )
        )
        + 1,
        int(
            checkpoint.get(
                "global_step",
                0,
            )
        ),
        float(
            checkpoint.get(
                "best_metric",
                float("inf"),
            )
        ),
    )


def make_rolling_buffers(
    window: int,
) -> Dict[str, Deque[float]]:
    return {
        "loss": deque(
            maxlen=max(1, int(window))
        ),
        "expr_loss": deque(
            maxlen=max(1, int(window))
        ),
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
    model: ScKITEStage1Model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    global_step: int,
    rolling_buffers: Dict[
        str,
        Deque[float],
    ],
    val_loader=None,
    run_dir: Optional[Path] = None,
    best_metric: float = float("inf"),
    history: Optional[
        List[Dict[str, Any]]
    ] = None,
) -> Tuple[
    int,
    Dict[str, float],
    float,
]:
    train_cfg = cfg.get("train", {})
    use_amp = (
        bool(
            train_cfg.get(
                "amp",
                True,
            )
        )
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

    model.train()
    loss_sum = 0.0
    n_steps = 0.0
    overflow_skips = 0

    progress = tqdm(
        loader,
        desc=f"train epoch {epoch}",
        leave=True,
        disable=not is_main_process(),
    )

    for batch in progress:
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
            _, loss = forward_one_batch(
                model,
                batch,
            )

        optimizer_was_run = True

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            scale_before = float(
                scaler.get_scale()
            )
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(
                scaler.get_scale()
            )
            optimizer_was_run = (
                scale_after >= scale_before
            )

            if not optimizer_was_run:
                overflow_skips += 1
                if is_main_process():
                    print(
                        "[AMP overflow] "
                        "optimizer step skipped; "
                        f"count={overflow_skips}"
                    )
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
            optimizer.step()

        if not optimizer_was_run:
            continue

        if scheduler is not None:
            scheduler.step()

        global_step += 1

        reduced = reduce_sum_tensor(
            [
                float(
                    loss.detach().item()
                )
            ],
            device=device,
        )
        step_loss = (
            reduced[0]
            / float(get_world_size())
        )

        rolling_buffers[
            "loss"
        ].append(step_loss)
        rolling_buffers[
            "expr_loss"
        ].append(step_loss)

        loss_sum += step_loss
        n_steps += 1.0

        rolling_metrics = {
            "train_loss":
                rolling_mean(
                    rolling_buffers["loss"]
                ),
            "train_expr_loss":
                rolling_mean(
                    rolling_buffers[
                        "expr_loss"
                    ]
                ),
            "learning_rate":
                float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
        }

        if (
            wandb_log_every > 0
            and global_step
            % wandb_log_every == 0
            and is_main_process()
            and wandb is not None
            and wandb.run is not None
        ):
            wandb.log(
                format_wandb_metrics(
                    {
                        **rolling_metrics,
                        "epoch": int(epoch),
                        "global_step":
                            int(global_step),
                    }
                ),
                step=global_step,
            )

        if (
            log_every > 0
            and global_step
            % log_every == 0
            and is_main_process()
        ):
            progress.set_postfix(
                {
                    "loss":
                        f"{rolling_metrics['train_loss']:.4f}",
                    "lr":
                        f"{rolling_metrics['learning_rate']:.2e}",
                }
            )

        if (
            val_loader is not None
            and eval_every_n_steps > 0
            and global_step
            % eval_every_n_steps == 0
        ):
            val_metrics = (
                evaluate_validation(
                    model,
                    val_loader,
                    device,
                    cfg,
                )
            )
            current_metric = float(
                val_metrics[
                    "val_expr_mse"
                ]
            )
            should_save_best = (
                math.isfinite(
                    current_metric
                )
                and current_metric
                < best_metric
            )
            if should_save_best:
                best_metric = current_metric

            if is_main_process():
                record = {
                    "epoch": int(epoch),
                    "global_step":
                        int(global_step),
                    **val_metrics,
                }
                print(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                if history is not None:
                    history.append(record)
                    if run_dir is not None:
                        save_json(
                            {"history": history},
                            run_dir
                            / "metrics_history.json",
                        )

                if (
                    wandb is not None
                    and wandb.run is not None
                ):
                    wandb.log(
                        format_wandb_metrics(
                            record
                        ),
                        step=global_step,
                    )

                if (
                    should_save_best
                    and run_dir is not None
                ):
                    save_checkpoint(
                        run_dir / "best.pt",
                        epoch,
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_metric,
                        record,
                        cfg,
                    )
                    print(
                        "saved best checkpoint "
                        "(minimum val_expr_mse="
                        f"{current_metric:.6f}) "
                        f"to {run_dir / 'best.pt'}"
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
                        record,
                        cfg,
                    )

            if is_dist_avail_and_initialized():
                dist.barrier()
            model.train()

    n_steps = max(n_steps, 1.0)
    return (
        global_step,
        {
            "train_loss":
                loss_sum / n_steps,
            "train_expr_loss":
                loss_sum / n_steps,
            "learning_rate":
                float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
        },
        best_metric,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train scKITE encoder-only baseline "
            "with masked-expression reconstruction. "
            "Only train and validation splits are used."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ[
        "STAGE2_CONFIG_PATH"
    ] = str(
        Path(args.config)
        .expanduser()
        .resolve()
    )

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    paths_cfg = cfg.get("paths", {})

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

    tokenizer = build_tokenizer(cfg)
    train_loader, val_loader = (
        build_loaders(
            cfg,
            tokenizer,
        )
    )

    if is_main_process():
        print(tokenizer)
        print(
            "[training mode] encoder-only"
        )
        print(
            "[objective] "
            "masked-expression reconstruction"
        )
        print(
            "[best checkpoint] "
            "minimize val_expr_mse"
        )

        run_dir = make_run_dir(
            paths_cfg.get(
                "output_root",
                "./runs",
            ),
            paths_cfg.get(
                "run_name",
                "encoder_only",
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
        objects = [
            str(run_dir)
            if is_main_process()
            else ""
        ]
        dist.broadcast_object_list(
            objects,
            src=0,
        )
        run_dir = Path(objects[0])

    model = build_model(
        cfg,
        tokenizer,
        device,
    )

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(
                train_cfg.get(
                    "find_unused_parameters",
                    False,
                )
            ),
        )

    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    n_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if is_main_process():
        print(f"device={device}")
        print(
            f"n_params={n_params:,}, "
            f"n_trainable={n_trainable:,}"
        )

    optimizer = AdamW(
        (
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ),
        lr=float(
            train_cfg.get(
                "lr",
                1e-4,
            )
        ),
        weight_decay=float(
            train_cfg.get(
                "weight_decay",
                0.01,
            )
        ),
    )

    epochs = int(
        train_cfg.get(
            "epochs",
            15,
        )
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(
        1,
        epochs * steps_per_epoch,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=float(
            train_cfg.get(
                "min_lr",
                1e-6,
            )
        ),
    )

    use_amp = (
        bool(
            train_cfg.get(
                "amp",
                True,
            )
        )
        and device.type == "cuda"
    )
    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
        if device.type == "cuda"
        else None
    )

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")

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

    wandb_cfg = cfg.get("wandb", {})
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
                "wandb 未安装，跳过记录。"
            )
        else:
            wandb.init(
                project=wandb_cfg.get(
                    "project",
                    "scKITE_encoder_only",
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

    history: List[Dict[str, Any]] = []
    keep_last_n = int(
        train_cfg.get(
            "keep_last_n_ckpts",
            1,
        )
    )
    save_every = int(
        train_cfg.get(
            "save_every_n_epochs",
            1,
        )
    )
    rolling_buffers = make_rolling_buffers(
        int(
            train_cfg.get(
                "wandb_loss_window",
                100,
            )
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
                epoch=epoch,
                global_step=global_step,
                rolling_buffers=
                    rolling_buffers,
                val_loader=val_loader,
                run_dir=run_dir,
                best_metric=best_metric,
                history=(
                    history
                    if is_main_process()
                    else None
                ),
            )

            val_metrics = evaluate_validation(
                model,
                val_loader,
                device,
                cfg,
            )
            current_metric = float(
                val_metrics["val_expr_mse"]
            )
            should_save_best = (
                math.isfinite(current_metric)
                and current_metric
                < best_metric
            )
            if should_save_best:
                best_metric = current_metric

            epoch_record = {
                "epoch": int(epoch),
                "global_step":
                    int(global_step),
                "epoch_train_summary":
                    train_metrics,
                **val_metrics,
            }

            if is_main_process():
                print(
                    json.dumps(
                        epoch_record,
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                history.append(epoch_record)
                save_json(
                    {"history": history},
                    run_dir
                    / "metrics_history.json",
                )

                if (
                    wandb is not None
                    and wandb.run is not None
                ):
                    wandb.log(
                        format_wandb_metrics(
                            val_metrics
                        ),
                        step=global_step,
                    )

                if should_save_best:
                    save_checkpoint(
                        run_dir / "best.pt",
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
                        "(minimum val_expr_mse="
                        f"{current_metric:.6f})"
                    )

                if (
                    save_every > 0
                    and (epoch + 1)
                    % save_every == 0
                ):
                    checkpoint_path = (
                        run_dir
                        / f"epoch_{epoch:04d}.pt"
                    )
                    save_checkpoint(
                        checkpoint_path,
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
