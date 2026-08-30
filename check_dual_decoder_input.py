#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 scKODE 双 Decoder 数据经过 data.py 和 tokenizer.py 后的真实模型输入。

适用结构
--------
Shared Encoder
├── Regulon Decoder：multi_regulon_compact，K=3
└── Annotation Decoder：plain_text

脚本直接调用当前项目的：
- train.build_tokenizer()
- train.build_loaders()
- data.Stage2Collator
- tokenizer.GlobalGeneTextTokenizer

因此检查到的序列与正式训练时使用同一套构建逻辑。

主要检查
--------
1. 一个 raw cell 在 task_sampling_mode=all 下是否正确展开成：
   - Regulon 样本
   - Annotation 样本
2. 两个任务是否共享完全相同的 Encoder 输入。
3. Encoder 的 <cls>、表达分箱、表达 mask、padding 是否正确。
4. Regulon Decoder 是否为：
   <bos> <task> regulon TF1 <gene_sep> TF2 <gene_sep> TF3
   <startofanswer>
   TF1 <arrow> targets...
   <regulon_sep>
   TF2 <arrow> targets...
   <regulon_sep>
   TF3 <arrow> targets...
5. selected_regulon_ids 是否来自该 cell 的 active_regulon_ids。
6. K 个 Regulon 是否无重复，TF 顺序是否与 block 顺序一致。
7. cell-specific targets 是否属于该 cell 的 raw expressed genes。
8. target 顺序是否保持 importance-sorted lookup 的前缀顺序。
9. Dynamic Target Budget 后长度是否 <= max_decoder_length。
10. Annotation Decoder 的任务前缀、label mask、正文和 <eos> 是否正确。
11. 可选：将构造后的 batch 真正送入双 Decoder 模型并检查输出/loss。

默认检查 train split、epoch=0，并优先寻找同时具有两个任务的 cell。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


REGULON_TASK_ID = 0
ANNOTATION_TASK_ID = 1
IGNORE_INDEX = -100


# =====================================================================
# CLI / basic utilities
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查双 Decoder 的 Encoder/Decoder 实际输入序列。"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="双 Decoder YAML 配置文件。",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="默认检查训练数据。",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Regulon cyclic_without_replacement 采样所使用的 epoch。",
    )
    parser.add_argument(
        "--num_cells",
        type=int,
        default=2,
        help="展示多少个 raw cell。",
    )
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=500,
        help="最多扫描多少个 raw cell，以寻找同时具备两个任务的样本。",
    )
    parser.add_argument(
        "--allow_single_task",
        action="store_true",
        help="允许只生成一个 Decoder task 的 cell；默认优先要求两个任务均存在。",
    )
    parser.add_argument(
        "--max_encoder_rows",
        type=int,
        default=60,
        help="终端中每个 cell 最多展示多少个 Encoder token。",
    )
    parser.add_argument(
        "--max_decoder_rows",
        type=int,
        default=160,
        help="终端中每个 Decoder task 最多展示多少个位置。",
    )
    parser.add_argument(
        "--max_targets_per_regulon",
        type=int,
        default=40,
        help="终端中每个 Regulon 最多展示多少个 target；TSV/JSON 保存全部。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dual_decoder_input_check",
        help="检查结果保存目录。",
    )
    parser.add_argument(
        "--run_model",
        action="store_true",
        help="额外执行一次双 Decoder forward；会占用较多显存。",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="--run_model 时使用的设备。",
    )
    parser.add_argument(
        "--debug_seed",
        type=int,
        default=20260728,
        help="固定本次检查中的训练 mask，便于复现。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML 根对象必须是 dict：{path}")
    return cfg


def add_project_paths(config_path: Path) -> None:
    candidates = [
        Path.cwd().resolve(),
        config_path.parent.resolve(),
        Path(__file__).resolve().parent,
    ]
    for candidate in candidates:
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def set_debug_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda，但 CUDA 不可用。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_tsv(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    headers: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(headers) + "\n")
        for row in rows:
            handle.write(
                "\t".join(str(row.get(header, "")) for header in headers)
                + "\n"
            )


def print_rows(
    rows: Sequence[Dict[str, Any]],
    headers: Sequence[str],
    max_rows: int,
    max_token_width: int = 36,
) -> None:
    display = list(rows[: max(0, int(max_rows))])
    widths: Dict[str, int] = {}
    for header in headers:
        width = max(
            len(header),
            max(
                (len(str(row.get(header, ""))) for row in display),
                default=0,
            ),
        )
        if "token" in header or header in {"gene", "input", "target"}:
            width = min(width, max_token_width)
        widths[header] = width

    def format_cell(header: str, value: Any) -> str:
        text = str(value)
        width = widths[header]
        if len(text) > width:
            text = text[: max(width - 3, 1)] + "..."
        return text.ljust(width)

    print(" | ".join(format_cell(h, h) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in display:
        print(" | ".join(format_cell(h, row.get(h, "")) for h in headers))

    if len(rows) > len(display):
        print(
            f"... 终端省略 {len(rows) - len(display)} 行；"
            "TSV/JSON 中保存全部内容。"
        )


# =====================================================================
# Token display / decoding
# =====================================================================

def pretty_token(tokenizer, token_id: int) -> str:
    token_id = int(token_id)
    try:
        values = tokenizer.ids_to_debug_tokens([token_id])
        if values:
            return str(values[0])
    except Exception:
        pass
    return str(
        tokenizer.global_id_to_token.get(
            token_id,
            f"<missing:{token_id}>",
        )
    )


def gene_name(tokenizer, token_id: int) -> str:
    token_id = int(token_id)
    return str(
        tokenizer.global_id_to_gene_symbol.get(
            token_id,
            pretty_token(tokenizer, token_id),
        )
    )


def decode_global_text(tokenizer, ids: Sequence[int]) -> str:
    bert_tokens: List[str] = []
    prefix = str(tokenizer.text_token_prefix)
    suffix = str(tokenizer.token_suffix)

    for token_id in ids:
        token_id = int(token_id)
        token = tokenizer.global_id_to_token.get(token_id, "")
        if not (
            token.startswith(prefix)
            and token.endswith(suffix)
        ):
            continue
        end = -len(suffix) if suffix else None
        bert_tokens.append(token[len(prefix):end])

    if not bert_tokens:
        return ""
    return tokenizer.text_tokenizer.convert_tokens_to_string(bert_tokens)


def reconstruct_target_ids(item: Dict[str, Any]) -> List[int]:
    """
    teacher forcing:
      decoder_input = [BOS] + target[:-1]
      labels        = target，prefix labels 被替换成 -100

    item 中通常已有 full_ids；同时从 decoder_input/labels 重建一次用于交叉检查。
    """
    decoder_input = [int(x) for x in item["decoder_input_ids"]]
    labels = [int(x) for x in item["decoder_labels"]]

    if not decoder_input:
        return []
    if len(decoder_input) != len(labels):
        raise ValueError(
            f"decoder_input 与 labels 长度不同："
            f"{len(decoder_input)} vs {len(labels)}"
        )

    last_target = labels[-1]
    if last_target == IGNORE_INDEX:
        raise ValueError("最后一个 decoder label 是 -100，无法重建 <eos>。")

    return decoder_input[1:] + [last_target]


def decoder_position_rows(
    tokenizer,
    item: Dict[str, Any],
) -> List[Dict[str, Any]]:
    decoder_input = [int(x) for x in item["decoder_input_ids"]]
    labels = [int(x) for x in item["decoder_labels"]]
    target_ids = [int(x) for x in item.get("full_ids", reconstruct_target_ids(item))]

    rows: List[Dict[str, Any]] = []
    for pos, (input_id, label_id, target_id) in enumerate(
        zip(decoder_input, labels, target_ids)
    ):
        rows.append(
            {
                "pos": pos,
                "input_id": input_id,
                "input_token": pretty_token(tokenizer, input_id),
                "target_id": target_id,
                "target_token": pretty_token(tokenizer, target_id),
                "label": label_id,
                "loss_used": label_id != IGNORE_INDEX,
            }
        )
    return rows


# =====================================================================
# Raw cell / Encoder inspection
# =====================================================================

def iter_raw_examples(dataset, limit: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    优先直接迭代 StreamingDataset。
    """
    iterator = iter(dataset)
    for index, example in enumerate(iterator):
        if index >= int(limit):
            break
        yield index, dict(example)


def encoder_rows(
    tokenizer,
    item: Dict[str, Any],
) -> List[Dict[str, Any]]:
    gene_ids = item["input_gene_ids"].detach().cpu()
    input_values = item["input_values"].detach().cpu()
    target_values = item["target_values"].detach().cpu()
    gene_masks = item["gene_masks"].detach().cpu().bool()
    expr_masks = item["expr_masks"].detach().cpu().bool()

    rows: List[Dict[str, Any]] = []
    for pos in range(gene_ids.numel()):
        gid = int(gene_ids[pos].item())
        rows.append(
            {
                "pos": pos,
                "gene_id": gid,
                "gene": pretty_token(tokenizer, gid),
                "input_value": float(input_values[pos].item()),
                "target_value": float(target_values[pos].item()),
                "expr_mask": bool(expr_masks[pos].item()),
                "gene_mask": bool(gene_masks[pos].item()),
            }
        )
    return rows


def check_encoder(
    tokenizer,
    item: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    max_length = int(data_cfg.get("max_encoder_length", 2049))
    cls_value = float(data_cfg.get("cls_value", -1.0))
    mask_value = float(data_cfg.get("mask_value", -3.0))
    mask_gene_input = bool(data_cfg.get("mask_gene_input", False))
    mask_expr_input = bool(data_cfg.get("mask_expr_input", True))

    gene_ids = item["input_gene_ids"].detach().cpu()
    target_gene_ids = item["target_gene_ids"].detach().cpu()
    input_values = item["input_values"].detach().cpu()
    target_values = item["target_values"].detach().cpu()
    expr_masks = item["expr_masks"].detach().cpu().bool()
    gene_masks = item["gene_masks"].detach().cpu().bool()

    checks: Dict[str, Any] = {
        "length": int(gene_ids.numel()),
        "length_within_limit": bool(gene_ids.numel() <= max_length),
        "first_token_is_cls": bool(
            gene_ids.numel() > 0
            and int(gene_ids[0].item()) == int(tokenizer.cls_token_id)
        ),
        "cls_input_value_correct": bool(
            gene_ids.numel() > 0
            and math.isclose(
                float(input_values[0].item()),
                cls_value,
                abs_tol=1e-6,
                rel_tol=0.0,
            )
        ),
        "cls_not_masked": bool(
            gene_ids.numel() > 0
            and not bool(expr_masks[0].item())
            and not bool(gene_masks[0].item())
        ),
        "masked_token_count": int(expr_masks.sum().item()),
        "observed_expr_mask_fraction": float(
            expr_masks.sum().item() / max(gene_ids.numel() - 1, 1)
        ),
    }

    if expr_masks.any():
        checks["masked_expr_uses_mask_value"] = bool(
            torch.isclose(
                input_values[expr_masks],
                torch.full_like(input_values[expr_masks], mask_value),
                atol=1e-6,
                rtol=0.0,
            ).all().item()
        )
        checks["masked_target_preserved"] = bool(
            not torch.isclose(
                target_values[expr_masks],
                torch.full_like(target_values[expr_masks], mask_value),
                atol=1e-6,
                rtol=0.0,
            ).any().item()
        )
    else:
        checks["masked_expr_uses_mask_value"] = bool(not mask_expr_input)
        checks["masked_target_preserved"] = True

    if not mask_gene_input:
        checks["gene_ids_unchanged"] = bool(
            torch.equal(gene_ids, target_gene_ids)
        )
        checks["gene_mask_all_false"] = bool(
            not gene_masks.any().item()
        )
    else:
        checks["gene_ids_unchanged"] = True
        checks["gene_mask_all_false"] = True

    bool_checks = [
        value
        for value in checks.values()
        if isinstance(value, bool)
    ]
    checks["passed"] = bool(all(bool_checks))
    return checks


# =====================================================================
# Decoder structure checks
# =====================================================================

def common_decoder_checks(
    tokenizer,
    item: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    max_decoder_length = int(
        cfg.get("decoder", {}).get("max_decoder_length", 512)
    )

    decoder_input = [int(x) for x in item["decoder_input_ids"]]
    labels = [int(x) for x in item["decoder_labels"]]
    attention = [int(x) for x in item["decoder_attention_mask"]]
    full_ids = [int(x) for x in item.get("full_ids", reconstruct_target_ids(item))]
    reconstructed = reconstruct_target_ids(item)

    try:
        start_index = full_ids.index(int(tokenizer.start_answer_token_id))
    except ValueError:
        start_index = -1

    prefix_mask_ok = (
        start_index >= 0
        and all(x == IGNORE_INDEX for x in labels[: start_index + 1])
    )
    answer_label_ok = (
        start_index >= 0
        and all(
            int(label) == int(target)
            for label, target in zip(
                labels[start_index + 1:],
                full_ids[start_index + 1:],
            )
        )
    )

    checks = {
        "decoder_length": len(decoder_input),
        "length_within_limit": len(decoder_input) <= max_decoder_length,
        "input_label_length_equal": len(decoder_input) == len(labels),
        "attention_length_equal": len(attention) == len(decoder_input),
        "attention_all_one_before_padding": all(x == 1 for x in attention),
        "decoder_input_starts_bos": bool(
            decoder_input
            and decoder_input[0] == int(tokenizer.bos_token_id)
        ),
        "target_starts_task": bool(
            full_ids
            and full_ids[0] == int(tokenizer.task_token_id)
        ),
        "target_ends_eos": bool(
            full_ids
            and full_ids[-1] == int(tokenizer.eos_token_id)
        ),
        "reconstructed_target_matches_full_ids": reconstructed == full_ids,
        "labels_mask_prefix_through_startanswer": prefix_mask_ok,
        "labels_supervise_answer_and_eos": answer_label_ok,
    }
    return checks


def check_regulon_item(
    tokenizer,
    collator,
    item: Dict[str, Any],
    active_ids: Sequence[int],
    raw_expressed_ids: Sequence[int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    checks = common_decoder_checks(tokenizer, item, cfg)

    selected = [int(x) for x in item.get("selected_regulon_ids", [])]
    query_tfs = [int(x) for x in item.get("query_tf_ids", [])]
    blocks = list(item.get("regulon_blocks", []))
    active_set = set(int(x) for x in active_ids)
    expressed_set = set(int(x) for x in raw_expressed_ids)

    expected_pack = tokenizer.build_multi_regulon_target_ids(
        regulon_blocks=blocks,
        query_tf_ids=query_tfs,
        task_name="regulon",
    )
    expected_target = [int(x) for x in expected_pack["target_ids"]]
    actual_target = [int(x) for x in item["full_ids"]]

    block_tfs = [int(block["tf_token_id"]) for block in blocks]
    block_ids = [int(block["regulon_id"]) for block in blocks]

    target_membership_ok = True
    target_order_ok = True
    target_gene_type_ok = True
    target_count_by_block: List[int] = []

    block_reports: List[Dict[str, Any]] = []
    for block in blocks:
        rid = int(block["regulon_id"])
        tf_id = int(block["tf_token_id"])
        targets = [int(x) for x in block["target_token_ids"]]
        target_count_by_block.append(len(targets))

        expected_cell_targets = collator._cell_specific_targets(
            rid,
            expressed_set,
        )

        target_membership_ok &= all(
            target in expressed_set
            for target in targets
        )
        target_gene_type_ok &= all(
            tokenizer.global_id_is_gene(target)
            for target in targets
        )
        target_order_ok &= (
            targets == expected_cell_targets[: len(targets)]
        )

        block_reports.append(
            {
                "regulon_id": rid,
                "tf_id": tf_id,
                "tf": gene_name(tokenizer, tf_id),
                "target_count": len(targets),
                "target_ids": targets,
                "targets": [gene_name(tokenizer, x) for x in targets],
                "all_targets_expressed": all(
                    x in expressed_set for x in targets
                ),
                "importance_order_prefix_ok": (
                    targets == expected_cell_targets[: len(targets)]
                ),
            }
        )

    configured_k = int(
        cfg.get("data", {}).get("regulon_num_queries", 3)
    )

    checks.update(
        {
            "selected_count": len(selected),
            "configured_k": configured_k,
            "selected_count_not_above_k": len(selected) <= configured_k,
            "selected_nonempty": len(selected) > 0,
            "selected_no_duplicates": len(selected) == len(set(selected)),
            "selected_subset_of_cell_active": set(selected).issubset(active_set),
            "block_ids_match_selected": block_ids == selected,
            "query_tfs_match_block_tfs": query_tfs == block_tfs,
            "query_tfs_no_duplicates": len(query_tfs) == len(set(query_tfs)),
            "each_block_has_targets": all(
                len(block["target_token_ids"]) > 0
                for block in blocks
            ),
            "all_targets_are_gene_tokens": target_gene_type_ok,
            "all_targets_in_raw_expressed_genes": target_membership_ok,
            "targets_preserve_importance_order": target_order_ok,
            "tokenizer_expected_target_exact_match": (
                actual_target == expected_target
            ),
            "target_counts_by_block": target_count_by_block,
            "total_target_gene_tokens": int(sum(target_count_by_block)),
        }
    )

    bool_checks = [
        value for value in checks.values()
        if isinstance(value, bool)
    ]
    checks["passed"] = bool(all(bool_checks))

    return {
        "checks": checks,
        "blocks": block_reports,
    }


def check_annotation_item(
    tokenizer,
    item: Dict[str, Any],
    raw_example: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    checks = common_decoder_checks(tokenizer, item, cfg)

    full_ids = [int(x) for x in item["full_ids"]]
    task_name_ids = tokenizer.encode_task_name_to_global_ids("annotation")
    expected_prefix = (
        [int(tokenizer.task_token_id)]
        + [int(x) for x in task_name_ids]
        + [int(tokenizer.start_answer_token_id)]
    )

    starts_with_expected_prefix = (
        full_ids[: len(expected_prefix)] == expected_prefix
    )
    body_ids = (
        full_ids[len(expected_prefix):-1]
        if starts_with_expected_prefix and len(full_ids) > len(expected_prefix)
        else []
    )
    decoded_text = decode_global_text(tokenizer, body_ids)

    annotation_field = "natural_language_annotation"
    for task_cfg in cfg.get("decoder_tasks", []):
        if (
            str(task_cfg.get("name", "")).lower() == "annotation"
            and task_cfg.get("enabled", True)
        ):
            annotation_field = str(
                task_cfg.get("field", annotation_field)
            )
            break

    raw_annotation = raw_example.get(annotation_field, "")
    raw_annotation_normalized = tokenizer.normalize_text(
        "" if raw_annotation is None else str(raw_annotation)
    )
    decoded_normalized = tokenizer.normalize_text(decoded_text)

    # Annotation 允许 tokenizer 截断，因此只要求：
    # 1. 前缀正确；
    # 2. 正文是文本 token；
    # 3. 解码结果非空；
    # 4. 未截断时与原始标准化文本一致；
    #    被截断时解码文本应为原始文本的前缀近似。
    body_all_text_tokens = all(
        tokenizer.global_id_is_text(token_id)
        for token_id in body_ids
    )
    exact_or_prefix_match = bool(
        decoded_normalized
        and (
            decoded_normalized == raw_annotation_normalized
            or raw_annotation_normalized.startswith(decoded_normalized)
        )
    )

    checks.update(
        {
            "annotation_prefix_correct": starts_with_expected_prefix,
            "annotation_body_nonempty": len(body_ids) > 0,
            "annotation_body_all_text_tokens": body_all_text_tokens,
            "decoded_annotation_nonempty": bool(decoded_normalized),
            "decoded_matches_raw_exact_or_prefix": exact_or_prefix_match,
            "raw_annotation_length_chars": len(raw_annotation_normalized),
            "decoded_annotation_length_chars": len(decoded_normalized),
        }
    )

    bool_checks = [
        value for value in checks.values()
        if isinstance(value, bool)
    ]
    checks["passed"] = bool(all(bool_checks))

    return {
        "checks": checks,
        "annotation_field": annotation_field,
        "raw_annotation": raw_annotation_normalized,
        "decoded_annotation": decoded_normalized,
        "body_ids": body_ids,
    }


# =====================================================================
# Display
# =====================================================================

def print_boolean_checks(checks: Dict[str, Any]) -> None:
    for key, value in checks.items():
        if isinstance(value, bool):
            print(f"  [{'PASS' if value else 'FAIL'}] {key}: {value}")


def display_regulon(
    tokenizer,
    item: Dict[str, Any],
    report: Dict[str, Any],
    max_decoder_rows: int,
    max_targets: int,
) -> None:
    print("\n" + "-" * 100)
    print("Regulon Decoder（task_id=0）")
    print("-" * 100)

    selected = item.get("selected_regulon_ids", [])
    query_tfs = item.get("query_tf_ids", [])

    print(f"selected_regulon_ids: {selected}")
    print(
        "query TFs: "
        + " | ".join(gene_name(tokenizer, x) for x in query_tfs)
    )
    print(
        "每个 block 的 target 数量: "
        f"{report['checks']['target_counts_by_block']}"
    )
    print(
        f"Decoder 总长度: {report['checks']['decoder_length']} | "
        f"target gene tokens: {report['checks']['total_target_gene_tokens']}"
    )

    print("\n模型看到的 compact Regulon target：")
    for block in report["blocks"]:
        targets = block["targets"]
        shown = targets[: max(0, int(max_targets))]
        suffix = (
            f" ...（另有 {len(targets) - len(shown)} 个）"
            if len(targets) > len(shown)
            else ""
        )
        print(
            f"  [Regulon {block['regulon_id']}] "
            f"{block['tf']} <arrow> "
            + " ".join(shown)
            + suffix
        )

    print("\n逐位置 teacher-forcing 输入与监督：")
    rows = decoder_position_rows(tokenizer, item)
    headers = [
        "pos",
        "input_id",
        "input_token",
        "target_id",
        "target_token",
        "label",
        "loss_used",
    ]
    print_rows(rows, headers, max_decoder_rows)
    print("\nRegulon 结构检查：")
    print_boolean_checks(report["checks"])


def display_annotation(
    tokenizer,
    item: Dict[str, Any],
    report: Dict[str, Any],
    max_decoder_rows: int,
) -> None:
    print("\n" + "-" * 100)
    print("Annotation Decoder（task_id=1）")
    print("-" * 100)
    print(f"原始 annotation：{report['raw_annotation']}")
    print(f"Tokenizer 解码正文：{report['decoded_annotation']}")
    print(f"Decoder 总长度: {report['checks']['decoder_length']}")

    print("\n逐位置 teacher-forcing 输入与监督：")
    rows = decoder_position_rows(tokenizer, item)
    headers = [
        "pos",
        "input_id",
        "input_token",
        "target_id",
        "target_token",
        "label",
        "loss_used",
    ]
    print_rows(rows, headers, max_decoder_rows)
    print("\nAnnotation 结构检查：")
    print_boolean_checks(report["checks"])


# =====================================================================
# Optional model forward
# =====================================================================

def run_model_forward(
    cfg: Dict[str, Any],
    tokenizer,
    collator,
    raw_example: Dict[str, Any],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    from train import (
        build_model,
        forward_one_batch,
        move_batch_to_device,
    )

    set_debug_seed(seed)
    batch = collator([dict(raw_example)])
    batch = move_batch_to_device(batch, device)

    model = build_model(cfg, tokenizer, device)
    model.eval()

    use_amp = (
        bool(cfg.get("train", {}).get("amp", True))
        and device.type == "cuda"
    )

    with torch.no_grad():
        if use_amp:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs, losses = forward_one_batch(
                    model,
                    batch,
                    cfg,
                )
        else:
            outputs, losses = forward_one_batch(
                model,
                batch,
                cfg,
            )

    result: Dict[str, Any] = {
        "device": str(device),
        "batch_size_after_task_expansion": int(
            batch["encoder_input_gene_ids"].shape[0]
        ),
        "encoder_input_shape": list(
            batch["encoder_input_gene_ids"].shape
        ),
        "decoder_input_shape": list(
            batch["decoder_input_ids"].shape
        ),
        "decoder_task_ids": batch["decoder_task_ids"].detach().cpu().tolist(),
        "encoder_outputs_shape": list(outputs["encoder_outputs"].shape),
        "expr_preds_shape": list(outputs["expr_preds"].shape),
        "regulon_batch_indices": (
            None
            if outputs.get("regulon_batch_indices") is None
            else outputs["regulon_batch_indices"].detach().cpu().tolist()
        ),
        "annotation_batch_indices": (
            None
            if outputs.get("annotation_batch_indices") is None
            else outputs["annotation_batch_indices"].detach().cpu().tolist()
        ),
        "regulon_logits_shape": (
            None
            if outputs.get("regulon_logits") is None
            else list(outputs["regulon_logits"].shape)
        ),
        "annotation_logits_shape": (
            None
            if outputs.get("annotation_logits") is None
            else list(outputs["annotation_logits"].shape)
        ),
        "loss": float(losses["loss"].detach().float().item()),
        "loss_expr": float(losses["loss_expr"].detach().float().item()),
        "loss_regulon": float(losses["loss_regulon"].detach().float().item()),
        "loss_annotation": float(
            losses["loss_annotation"].detach().float().item()
        ),
    }

    finite_values = [
        result["loss"],
        result["loss_expr"],
        result["loss_regulon"],
        result["loss_annotation"],
    ]
    result["all_losses_finite"] = bool(
        all(math.isfinite(x) for x in finite_values)
    )
    result["passed"] = result["all_losses_finite"]

    del outputs, losses, batch, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["STAGE2_CONFIG_PATH"] = str(config_path)
    add_project_paths(config_path)
    cfg = load_yaml(config_path)

    from train import build_tokenizer, build_loaders

    tokenizer = build_tokenizer(cfg)
    train_loader, val_loader = build_loaders(cfg, tokenizer)
    loader = train_loader if args.split == "train" else val_loader
    dataset = loader.dataset
    collator = loader.collate_fn

    if hasattr(collator, "set_epoch"):
        collator.set_epoch(int(args.epoch))

    print("=" * 100)
    print("双 Decoder 输入检查")
    print("=" * 100)
    print(tokenizer)
    print(f"config: {config_path}")
    print(f"split: {args.split}")
    print(f"epoch: {args.epoch}")
    print(
        "task_sampling_mode: "
        f"{cfg.get('data', {}).get('task_sampling_mode', 'all')}"
    )
    print(
        "regulon_num_queries: "
        f"{cfg.get('data', {}).get('regulon_num_queries', 3)}"
    )
    print(
        "max_encoder_length: "
        f"{cfg.get('data', {}).get('max_encoder_length', 2049)}"
    )
    print(
        "max_decoder_length: "
        f"{cfg.get('decoder', {}).get('max_decoder_length', 512)}"
    )

    selected_cells: List[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]] = []

    for raw_index, raw_example in iter_raw_examples(
        dataset,
        args.scan_limit,
    ):
        # 每个 raw cell 使用固定 debug seed，使本次展示可复现。
        cell_seed = int(args.debug_seed) + int(raw_index)
        set_debug_seed(cell_seed)

        try:
            items = collator._prepare_one(dict(raw_example))
        except Exception as exc:
            print(
                f"[跳过 raw index={raw_index}] "
                f"构建失败：{type(exc).__name__}: {exc}"
            )
            continue

        if not items:
            continue

        task_ids = {
            int(item["decoder_task_id"])
            for item in items
        }
        has_both = (
            REGULON_TASK_ID in task_ids
            and ANNOTATION_TASK_ID in task_ids
        )

        if not args.allow_single_task and not has_both:
            continue

        selected_cells.append(
            (raw_index, dict(raw_example), items)
        )
        if len(selected_cells) >= int(args.num_cells):
            break

    if not selected_cells:
        raise RuntimeError(
            "未找到可检查的 cell。可增加 --scan_limit，"
            "或使用 --allow_single_task。"
        )

    all_reports: List[Dict[str, Any]] = []

    for display_index, (
        raw_index,
        raw_example,
        items,
    ) in enumerate(selected_cells):
        cell_key = str(items[0].get("cell_key", f"raw_{raw_index}"))

        mapped_genes, mapped_exprs = collator._extract_raw_mapped_gene_expr(
            raw_example
        )
        raw_expressed_set = collator._raw_expressed_gene_set(
            mapped_genes,
            mapped_exprs,
        )
        raw_expressed_ids = sorted(int(x) for x in raw_expressed_set)
        active_ids = collator._parse_active_regulon_ids(raw_example)

        task_counter = Counter(
            int(item["decoder_task_id"])
            for item in items
        )

        print("\n" + "=" * 100)
        print(
            f"Cell {display_index} | raw_index={raw_index} | "
            f"cell_key={cell_key}"
        )
        print("=" * 100)
        print(f"raw mapped genes: {len(mapped_genes)}")
        print(f"raw expressed genes: {len(raw_expressed_ids)}")
        print(f"active regulons: {len(active_ids)}")
        print(f"active_regulon_ids 前30个: {active_ids[:30]}")
        print(
            "task expansion: "
            f"regulon={task_counter.get(REGULON_TASK_ID, 0)}, "
            f"annotation={task_counter.get(ANNOTATION_TASK_ID, 0)}, "
            f"total={len(items)}"
        )

        # 两个 task 应共享相同 Encoder 输入。
        encoder_shared = True
        reference_item = items[0]
        for item in items[1:]:
            for key in [
                "input_gene_ids",
                "input_values",
                "target_gene_ids",
                "target_values",
                "gene_masks",
                "expr_masks",
            ]:
                if not torch.equal(reference_item[key], item[key]):
                    encoder_shared = False

        enc_rows = encoder_rows(tokenizer, reference_item)
        enc_checks = check_encoder(tokenizer, reference_item, cfg)
        enc_checks["encoder_identical_across_tasks"] = encoder_shared
        enc_checks["passed"] = bool(
            enc_checks["passed"] and encoder_shared
        )

        print("\nEncoder 实际输入序列：")
        print(
            f"有效长度={enc_checks['length']} | "
            f"表达mask={enc_checks['masked_token_count']} | "
            f"mask比例={enc_checks['observed_expr_mask_fraction']:.4f}"
        )
        print_rows(
            enc_rows,
            [
                "pos",
                "gene_id",
                "gene",
                "input_value",
                "target_value",
                "expr_mask",
                "gene_mask",
            ],
            args.max_encoder_rows,
        )
        print("\nEncoder 检查：")
        print_boolean_checks(enc_checks)

        cell_dir = output_dir / f"cell_{display_index:03d}_{cell_key}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        write_tsv(
            enc_rows,
            cell_dir / "encoder_input.tsv",
            [
                "pos",
                "gene_id",
                "gene",
                "input_value",
                "target_value",
                "expr_mask",
                "gene_mask",
            ],
        )

        task_reports: List[Dict[str, Any]] = []

        for task_order, item in enumerate(items):
            task_id = int(item["decoder_task_id"])

            if task_id == REGULON_TASK_ID:
                reg_report = check_regulon_item(
                    tokenizer=tokenizer,
                    collator=collator,
                    item=item,
                    active_ids=active_ids,
                    raw_expressed_ids=raw_expressed_ids,
                    cfg=cfg,
                )
                display_regulon(
                    tokenizer,
                    item,
                    reg_report,
                    args.max_decoder_rows,
                    args.max_targets_per_regulon,
                )
                rows = decoder_position_rows(tokenizer, item)
                write_tsv(
                    rows,
                    cell_dir / "regulon_decoder.tsv",
                    [
                        "pos",
                        "input_id",
                        "input_token",
                        "target_id",
                        "target_token",
                        "label",
                        "loss_used",
                    ],
                )
                write_json(
                    {
                        "item_metadata": {
                            "selected_regulon_ids": item.get(
                                "selected_regulon_ids", []
                            ),
                            "query_tf_ids": item.get("query_tf_ids", []),
                            "regulon_blocks": item.get("regulon_blocks", []),
                        },
                        **reg_report,
                    },
                    cell_dir / "regulon_report.json",
                )
                task_reports.append(
                    {
                        "task": "regulon",
                        "passed": reg_report["checks"]["passed"],
                        "report": reg_report,
                    }
                )

            elif task_id == ANNOTATION_TASK_ID:
                ann_report = check_annotation_item(
                    tokenizer=tokenizer,
                    item=item,
                    raw_example=raw_example,
                    cfg=cfg,
                )
                display_annotation(
                    tokenizer,
                    item,
                    ann_report,
                    args.max_decoder_rows,
                )
                rows = decoder_position_rows(tokenizer, item)
                write_tsv(
                    rows,
                    cell_dir / "annotation_decoder.tsv",
                    [
                        "pos",
                        "input_id",
                        "input_token",
                        "target_id",
                        "target_token",
                        "label",
                        "loss_used",
                    ],
                )
                write_json(
                    ann_report,
                    cell_dir / "annotation_report.json",
                )
                task_reports.append(
                    {
                        "task": "annotation",
                        "passed": ann_report["checks"]["passed"],
                        "report": ann_report,
                    }
                )
            else:
                raise ValueError(f"未知 decoder_task_id={task_id}")

        model_report: Optional[Dict[str, Any]] = None
        if args.run_model and display_index == 0:
            print("\n" + "-" * 100)
            print("执行双 Decoder 模型 forward")
            print("-" * 100)
            try:
                model_report = run_model_forward(
                    cfg=cfg,
                    tokenizer=tokenizer,
                    collator=collator,
                    raw_example=raw_example,
                    device=choose_device(args.device),
                    seed=int(args.debug_seed) + int(raw_index),
                )
                print(
                    json.dumps(
                        model_report,
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            except RuntimeError as exc:
                model_report = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    "模型 forward 失败。若为 CUDA OOM，"
                    "不影响前面的 data/tokenizer 结构检查。\n"
                    f"{model_report['error']}"
                )

        cell_report = {
            "display_index": display_index,
            "raw_index": raw_index,
            "cell_key": cell_key,
            "raw_mapped_gene_count": len(mapped_genes),
            "raw_expressed_gene_count": len(raw_expressed_ids),
            "active_regulon_count": len(active_ids),
            "active_regulon_ids": active_ids,
            "task_count": len(items),
            "encoder_checks": enc_checks,
            "tasks": task_reports,
            "model_forward": model_report,
        }
        write_json(cell_report, cell_dir / "cell_summary.json")
        all_reports.append(cell_report)

    all_data_checks_passed = bool(
        all(
            cell["encoder_checks"]["passed"]
            and all(task["passed"] for task in cell["tasks"])
            for cell in all_reports
        )
    )

    model_reports = [
        cell["model_forward"]
        for cell in all_reports
        if cell["model_forward"] is not None
    ]
    all_model_checks_passed = (
        None
        if not model_reports
        else bool(all(report.get("passed", False) for report in model_reports))
    )

    final_report = {
        "config": str(config_path),
        "split": args.split,
        "epoch": int(args.epoch),
        "num_cells_checked": len(all_reports),
        "all_data_tokenizer_checks_passed": all_data_checks_passed,
        "all_model_checks_passed": all_model_checks_passed,
        "overall_passed": bool(
            all_data_checks_passed
            and (
                all_model_checks_passed
                if all_model_checks_passed is not None
                else True
            )
        ),
        "cells": all_reports,
    }
    write_json(final_report, output_dir / "check_summary.json")

    print("\n" + "=" * 100)
    print("最终检查结果")
    print("=" * 100)
    print(
        "all_data_tokenizer_checks_passed: "
        f"{all_data_checks_passed}"
    )
    print(
        "all_model_checks_passed: "
        f"{all_model_checks_passed}"
    )
    print(f"overall_passed: {final_report['overall_passed']}")
    print(f"输出目录: {output_dir}")
    print(f"总报告: {output_dir / 'check_summary.json'}")


if __name__ == "__main__":
    main()
