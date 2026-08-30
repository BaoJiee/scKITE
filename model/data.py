#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import random
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

try:
    from streaming import StreamingDataset
except Exception as exc:
    raise ImportError(
        "无法导入 streaming.StreamingDataset，请先安装 mosaicml-streaming / streaming。"
    ) from exc

from tokenizer import GlobalGeneTextTokenizer


PathLike = Union[str, Path]
REGULON_TASK_ID = 0
ANNOTATION_TASK_ID = 1


# =====================================================================
# Encoder utilities
# =====================================================================

def left_binning(values: torch.Tensor, num_bins: int) -> torch.Tensor:
    if values.numel() == 0:
        return values
    out = torch.zeros_like(values, dtype=torch.long)
    pos_mask = values > 0
    if pos_mask.sum() == 0:
        return out
    pos_values = values[pos_mask].float()
    if pos_values.numel() == 1:
        out[pos_mask] = 1
        return out
    q = torch.linspace(0.0, 1.0, steps=int(num_bins), device=values.device)
    edges = torch.quantile(pos_values, q)
    inner_edges = edges[1:-1].contiguous()
    out[pos_mask] = torch.bucketize(pos_values, inner_edges, right=False).long() + 1
    return out


def sample_or_truncate(
    genes: torch.Tensor,
    exprs: torch.Tensor,
    max_length: int,
    keep_first_n_tokens: int = 1,
    sampling: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert genes.ndim == 1 and exprs.ndim == 1
    assert genes.shape[0] == exprs.shape[0]
    if genes.shape[0] <= int(max_length):
        return genes, exprs

    prefix_genes = genes[:keep_first_n_tokens]
    prefix_exprs = exprs[:keep_first_n_tokens]
    body_genes = genes[keep_first_n_tokens:]
    body_exprs = exprs[keep_first_n_tokens:]
    need = int(max_length) - int(keep_first_n_tokens)
    if need <= 0:
        return prefix_genes[:max_length], prefix_exprs[:max_length]

    if sampling:
        idx = torch.randperm(body_genes.shape[0], generator=generator)[:need]
        idx, _ = torch.sort(idx)
    else:
        idx = torch.arange(need)

    return (
        torch.cat([prefix_genes, body_genes[idx]], dim=0),
        torch.cat([prefix_exprs, body_exprs[idx]], dim=0),
    )


def parse_json_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if ((text.startswith("[") and text.endswith("]")) or
                (text.startswith("{") and text.endswith("}"))):
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def read_json_or_jsonl(path: PathLike) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"文件为空：{path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        out = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSONL：{line[:160]}") from exc
        return out


def _stable_crc32(text: str) -> int:
    return int(zlib.crc32(str(text).encode("utf-8")) & 0xFFFFFFFF)


# =====================================================================
# Regulon lookup
# =====================================================================

def load_regulon_target_lookup(
    path: PathLike,
    n_regulons: int,
) -> Dict[int, Dict[str, Any]]:
    obj = read_json_or_jsonl(path)

    if isinstance(obj, dict):
        for wrapper_key in ("regulons", "items", "data"):
            if wrapper_key in obj and isinstance(obj[wrapper_key], (dict, list)):
                obj = obj[wrapper_key]
                break

    lookup: Dict[int, Dict[str, Any]] = {}
    if isinstance(obj, dict):
        for key, rec in obj.items():
            if not isinstance(rec, dict):
                continue
            rid = int(rec.get("regulon_id", key))
            lookup[rid] = dict(rec)
    elif isinstance(obj, list):
        for idx, rec in enumerate(obj):
            if not isinstance(rec, dict):
                continue
            rid = int(rec.get("regulon_id", rec.get("id", idx)))
            lookup[rid] = dict(rec)
    else:
        raise TypeError(f"Regulon target 根对象必须为 dict/list，当前为 {type(obj).__name__}")

    expected = set(range(int(n_regulons)))
    actual = set(lookup.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Regulon ID 必须完整对应 0..{n_regulons - 1}。"
            f" missing={missing[:20]}, extra={extra[:20]}"
        )

    normalized: Dict[int, Dict[str, Any]] = {}
    for rid in range(int(n_regulons)):
        rec = dict(lookup[rid])
        if "tf_token_id" not in rec:
            raise KeyError(f"Regulon {rid} 缺少 tf_token_id。")
        targets = rec.get("target_token_ids", [])
        if not isinstance(targets, (list, tuple)):
            raise TypeError(f"Regulon {rid} 的 target_token_ids 必须为 list。")
        rec["regulon_id"] = rid
        rec["tf_token_id"] = int(rec["tf_token_id"])
        rec["target_token_ids"] = [int(x) for x in targets]
        if "target_importances" in rec and rec["target_importances"] is not None:
            imps = list(rec["target_importances"])
            if len(imps) != len(rec["target_token_ids"]):
                raise ValueError(
                    f"Regulon {rid}: target_importances 与 target_token_ids 长度不一致。"
                )
            rec["target_importances"] = [float(x) for x in imps]
        normalized[rid] = rec
    return normalized


# =====================================================================
# Stage2 collator
# =====================================================================

class Stage2Collator:
    """
    新版 Stage2 collator。

    - Regulon Decoder：K 个 active regulons，compact multi-regulon format
    - Annotation Decoder：plain text
    - Train：epoch-aware cyclic selection
    - Validation：fixed deterministic selection
    - Cell-specific target：global targets ∩ raw expressed genes
    - Dynamic Target Budget：按 importance 顺序、round-robin 分配 target token budget
    """

    def __init__(
        self,
        tokenizer: GlobalGeneTextTokenizer,
        max_encoder_length: int = 2049,
        max_decoder_length: int = 1025,
        mlm_probability: float = 0.10,
        num_bins: int = 51,
        sampling: bool = False,
        keep_first_n_tokens: int = 1,
        pad_value: float = -2.0,
        cls_value: float = -1.0,
        mask_value: float = -3.0,
        mask_gene_input: bool = False,
        mask_expr_input: bool = True,
        decoder_tasks: Optional[List[Dict[str, Any]]] = None,
        genes_field: str = "genes",
        expressions_field: str = "expressions",
        loss_weight_field: Optional[str] = "loss_weight",
        use_loss_weight: bool = False,
        task_sampling_mode: str = "all",
        regulon_target_path: Optional[PathLike] = None,
        active_regulon_field: str = "active_regulon_ids",
        cell_id_field: str = "cell_id",
        n_regulons: int = 530,
        regulon_num_queries: int = 3,
        regulon_sampling_mode: str = "cyclic_without_replacement",
        regulon_seed: int = 42,
        validation_regulon_seed: int = 2026,
        cell_specific_targets: bool = True,
        expressed_gene_threshold: float = 0.0,
        dynamic_target_budget: bool = True,
        is_training: bool = True,
        fixed_eval_encoder_mask: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_encoder_length = int(max_encoder_length)
        self.max_decoder_length = int(max_decoder_length)
        self.mlm_probability = float(mlm_probability)
        self.num_bins = int(num_bins)
        self.sampling = bool(sampling)
        self.keep_first_n_tokens = int(keep_first_n_tokens)
        self.pad_value = float(pad_value)
        self.cls_value = float(cls_value)
        self.mask_value = float(mask_value)
        self.mask_gene_input = bool(mask_gene_input)
        self.mask_expr_input = bool(mask_expr_input)

        self.genes_field = str(genes_field)
        self.expressions_field = str(expressions_field)
        self.loss_weight_field = loss_weight_field
        self.use_loss_weight = bool(use_loss_weight)

        self.active_regulon_field = str(active_regulon_field)
        self.cell_id_field = str(cell_id_field)
        self.n_regulons = int(n_regulons)
        self.regulon_num_queries = int(regulon_num_queries)
        self.regulon_sampling_mode = str(regulon_sampling_mode).lower().strip()
        self.regulon_seed = int(regulon_seed)
        self.validation_regulon_seed = int(validation_regulon_seed)
        self.cell_specific_targets = bool(cell_specific_targets)
        self.expressed_gene_threshold = float(expressed_gene_threshold)
        self.dynamic_target_budget = bool(dynamic_target_budget)
        self.is_training = bool(is_training)
        self.fixed_eval_encoder_mask = bool(fixed_eval_encoder_mask)
        self.epoch = 0

        if self.n_regulons <= 0:
            raise ValueError("n_regulons 必须 > 0。")
        if self.regulon_num_queries <= 0:
            raise ValueError("regulon_num_queries 必须 > 0。")
        if self.regulon_sampling_mode not in {
            "cyclic_without_replacement",
            "fixed",
            "random_without_replacement",
        }:
            raise ValueError(
                "regulon_sampling_mode 仅支持 cyclic_without_replacement / fixed / random_without_replacement。"
            )

        if regulon_target_path is None:
            raise ValueError("必须提供 regulon_target_path。")
        self.regulon_target_path = str(regulon_target_path)
        self.regulon_lookup = load_regulon_target_lookup(
            self.regulon_target_path,
            n_regulons=self.n_regulons,
        )

        self.decoder_tasks = list(
            decoder_tasks
            or [
                {
                    "name": "regulon",
                    "field": self.active_regulon_field,
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
        )
        self.task_sampling_mode = str(task_sampling_mode).lower().strip()
        if self.task_sampling_mode not in {"weighted", "all"}:
            raise ValueError("task_sampling_mode 仅支持 'weighted' 或 'all'。")

        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.cls_token_id = int(self.tokenizer.cls_token_id)
        self.mask_gene_token_id = int(self.tokenizer.mask_gene_token_id)
        self._validate_decoder_tasks()

    # -----------------------------------------------------------------
    # Epoch / stable key
    # -----------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _stable_cell_key(self, example: Dict[str, Any]) -> str:
        for key in [self.cell_id_field, "cell_id", "barcode", "obs_names", "index"]:
            if key and key in example:
                value = example[key]
                if value is not None and str(value).strip():
                    return str(value)

        genes = parse_json_if_needed(example.get(self.genes_field, []))
        exprs = parse_json_if_needed(example.get(self.expressions_field, []))
        try:
            gene_preview = list(genes)[:64]
            expr_preview = [float(x) for x in list(exprs)[:64]]
        except Exception:
            gene_preview, expr_preview = [], []
        return f"fallback:{_stable_crc32(json.dumps([gene_preview, expr_preview], default=str))}"

    def _make_generator(self, key: str, salt: int) -> torch.Generator:
        seed = (_stable_crc32(f"{key}|{salt}") + int(salt)) % (2**31 - 1)
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    # -----------------------------------------------------------------
    # Decoder task config
    # -----------------------------------------------------------------

    @staticmethod
    def _get_task_id(task_cfg: Dict[str, Any]) -> int:
        name = str(task_cfg.get("name", "")).lower().strip()
        target_type = str(task_cfg.get("target_type", "")).lower().strip()
        if name == "regulon" and target_type in {
            "multi_regulon_compact",
            "gene_pairs",
            "regulon_edges",
        }:
            return REGULON_TASK_ID
        if name == "annotation" and target_type == "plain_text":
            return ANNOTATION_TASK_ID
        raise ValueError(
            "当前仅支持 regulon/multi_regulon_compact 与 annotation/plain_text。"
            f" got name={name!r}, target_type={target_type!r}"
        )

    def _validate_decoder_tasks(self) -> None:
        seen = set()
        for cfg in self.decoder_tasks:
            if not cfg.get("enabled", True):
                continue
            tid = self._get_task_id(cfg)
            if tid in seen:
                raise ValueError(f"Decoder task 重复配置：task_id={tid}")
            seen.add(tid)
        if not seen:
            raise ValueError("没有启用任何 decoder task。")

    # -----------------------------------------------------------------
    # Raw genes / expression
    # -----------------------------------------------------------------

    def _map_gene_value_to_global_id(self, gene_value: Any) -> Optional[int]:
        if gene_value is None:
            return None
        if torch.is_tensor(gene_value):
            gene_value = gene_value.item()
        if hasattr(gene_value, "item") and not isinstance(gene_value, (str, bytes)):
            try:
                gene_value = gene_value.item()
            except Exception:
                pass

        if not isinstance(gene_value, bool):
            try:
                if isinstance(gene_value, (int, float)) or str(gene_value).strip().isdigit():
                    gid = int(gene_value)
                    if gid in self.tokenizer.global_id_to_token and self.tokenizer.global_id_is_gene(gid):
                        return gid
            except Exception:
                pass
        return self.tokenizer.gene_name_to_global_id(str(gene_value))

    def _extract_raw_mapped_gene_expr(
        self,
        example: Dict[str, Any],
    ) -> Tuple[List[int], List[float]]:
        if self.genes_field not in example:
            raise KeyError(f"样本缺少 genes 字段 {self.genes_field!r}。")
        if self.expressions_field not in example:
            raise KeyError(f"样本缺少 expressions 字段 {self.expressions_field!r}。")

        raw_genes = list(parse_json_if_needed(example[self.genes_field]))
        raw_exprs = list(parse_json_if_needed(example[self.expressions_field]))
        if len(raw_genes) != len(raw_exprs):
            raise ValueError(f"genes 与 expressions 长度不一致：{len(raw_genes)} vs {len(raw_exprs)}")

        mapped_genes: List[int] = []
        mapped_exprs: List[float] = []
        for gene, expr in zip(raw_genes, raw_exprs):
            gid = self._map_gene_value_to_global_id(gene)
            if gid is None:
                continue
            mapped_genes.append(int(gid))
            mapped_exprs.append(float(expr))
        return mapped_genes, mapped_exprs

    def _raw_expressed_gene_set(
        self,
        mapped_genes: Sequence[int],
        mapped_exprs: Sequence[float],
    ) -> set:
        return {
            int(gid)
            for gid, expr in zip(mapped_genes, mapped_exprs)
            if float(expr) > self.expressed_gene_threshold
        }

    def _prepare_gene_expr_from_mapped(
        self,
        mapped_genes: Sequence[int],
        mapped_exprs: Sequence[float],
        cell_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        genes = torch.as_tensor(list(mapped_genes), dtype=torch.long)
        exprs = torch.as_tensor(list(mapped_exprs), dtype=torch.float)
        genes = torch.cat([torch.tensor([self.cls_token_id], dtype=torch.long), genes], dim=0)
        exprs = torch.cat([torch.tensor([self.cls_value], dtype=torch.float), exprs], dim=0)

        generator = None
        if self.sampling and (not self.is_training) and self.fixed_eval_encoder_mask:
            generator = self._make_generator(cell_key, 99173)

        return sample_or_truncate(
            genes=genes,
            exprs=exprs,
            max_length=self.max_encoder_length,
            keep_first_n_tokens=self.keep_first_n_tokens,
            sampling=self.sampling,
            generator=generator,
        )

    def _apply_encoder_mask(
        self,
        genes: torch.Tensor,
        exprs: torch.Tensor,
        cell_key: str,
    ) -> Dict[str, torch.Tensor]:
        target_gene_ids = genes.clone()
        binned_values = exprs.clone()
        if binned_values.shape[0] > self.keep_first_n_tokens:
            binned_values[self.keep_first_n_tokens:] = left_binning(
                binned_values[self.keep_first_n_tokens:], self.num_bins
            ).float()

        target_values = binned_values.clone()
        input_gene_ids = genes.clone()
        input_values = binned_values.clone()

        valid_mask = torch.zeros_like(genes, dtype=torch.bool)
        valid_mask[self.keep_first_n_tokens:] = True
        valid_mask &= genes.ne(self.pad_token_id)

        if self.mlm_probability <= 0:
            rand_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        else:
            generator = None
            if (not self.is_training) and self.fixed_eval_encoder_mask:
                generator = self._make_generator(cell_key, 314159)
            rand = torch.rand(genes.shape, dtype=torch.float, generator=generator)
            rand_mask = rand.lt(self.mlm_probability) & valid_mask
            if valid_mask.any() and not rand_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                if generator is None:
                    j = torch.randint(0, valid_indices.numel(), (1,)).item()
                else:
                    j = torch.randint(0, valid_indices.numel(), (1,), generator=generator).item()
                rand_mask[valid_indices[j]] = True

        if self.mask_gene_input:
            input_gene_ids[rand_mask] = self.mask_gene_token_id
            gene_masks = rand_mask.clone()
        else:
            gene_masks = torch.zeros_like(rand_mask, dtype=torch.bool)

        if self.mask_expr_input:
            input_values[rand_mask] = self.mask_value
            expr_masks = rand_mask.clone()
        else:
            expr_masks = torch.zeros_like(rand_mask, dtype=torch.bool)

        return {
            "input_gene_ids": input_gene_ids,
            "input_values": input_values,
            "target_gene_ids": target_gene_ids,
            "target_values": target_values,
            "gene_masks": gene_masks,
            "expr_masks": expr_masks,
        }

    # -----------------------------------------------------------------
    # Active Regulon IDs
    # -----------------------------------------------------------------

    def _parse_active_regulon_ids(self, example: Dict[str, Any]) -> List[int]:
        value = parse_json_if_needed(example.get(self.active_regulon_field, []))
        if value is None:
            return []
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict, str)):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            # 兼容 "1,2,3"
            value = [x.strip() for x in text.split(",") if x.strip()]
        if isinstance(value, dict):
            value = list(value.keys())
        if not isinstance(value, (list, tuple, set)):
            value = [value]

        ids: List[int] = []
        seen = set()
        for x in value:
            try:
                rid = int(x)
            except Exception:
                continue
            if 0 <= rid < self.n_regulons and rid not in seen:
                ids.append(rid)
                seen.add(rid)
        return ids

    # -----------------------------------------------------------------
    # Cell-specific regulon blocks
    # -----------------------------------------------------------------

    def _cell_specific_targets(self, rid: int, expressed_gene_set: set) -> List[int]:
        global_targets = self.regulon_lookup[int(rid)]["target_token_ids"]
        if not self.cell_specific_targets:
            return [int(x) for x in global_targets]
        # Preserve global importance order.
        return [int(x) for x in global_targets if int(x) in expressed_gene_set]

    def _eligible_active_regulons(
        self,
        active_ids: Sequence[int],
        expressed_gene_set: set,
    ) -> Tuple[List[int], Dict[int, List[int]]]:
        eligible: List[int] = []
        target_map: Dict[int, List[int]] = {}
        for rid in active_ids:
            if rid not in self.regulon_lookup:
                continue
            targets = self._cell_specific_targets(rid, expressed_gene_set)
            if not targets:
                continue
            eligible.append(int(rid))
            target_map[int(rid)] = targets
        return eligible, target_map

    def _stable_permutation(self, ids: Sequence[int], cell_key: str, seed: int) -> List[int]:
        ids = list(ids)
        rnd = random.Random(_stable_crc32(f"{cell_key}|{seed}"))
        rnd.shuffle(ids)
        return ids

    def _select_regulon_ids(
        self,
        eligible_ids: Sequence[int],
        cell_key: str,
    ) -> List[int]:
        eligible_ids = list(eligible_ids)
        if not eligible_ids:
            return []
        k = min(self.regulon_num_queries, len(eligible_ids))

        if not self.is_training:
            perm = self._stable_permutation(
                eligible_ids,
                cell_key,
                self.validation_regulon_seed,
            )
            return perm[:k]

        if self.regulon_sampling_mode == "fixed":
            perm = self._stable_permutation(eligible_ids, cell_key, self.regulon_seed)
            return perm[:k]

        if self.regulon_sampling_mode == "random_without_replacement":
            return random.sample(eligible_ids, k=k)

        # cyclic_without_replacement:
        # stable shuffle + epoch-dependent circular slice. No duplicates inside one selection.
        perm = self._stable_permutation(eligible_ids, cell_key, self.regulon_seed)
        n = len(perm)
        start = (int(self.epoch) * k) % n
        selected: List[int] = []
        cursor = start
        while len(selected) < k:
            rid = perm[cursor % n]
            if rid not in selected:
                selected.append(rid)
            cursor += 1
        return selected

    def _allocate_dynamic_target_budget(
        self,
        selected_ids: Sequence[int],
        target_map: Dict[int, List[int]],
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        selected_ids = list(selected_ids)
        while selected_ids:
            query_tf_ids = [int(self.regulon_lookup[rid]["tf_token_id"]) for rid in selected_ids]
            fixed = self.tokenizer.multi_regulon_fixed_token_count(query_tf_ids, task_name="regulon")
            budget = self.max_decoder_length - fixed
            if budget >= len(selected_ids):
                break
            selected_ids = selected_ids[:-1]

        if not selected_ids:
            return [], []

        query_tf_ids = [int(self.regulon_lookup[rid]["tf_token_id"]) for rid in selected_ids]
        fixed = self.tokenizer.multi_regulon_fixed_token_count(query_tf_ids, task_name="regulon")
        target_budget = self.max_decoder_length - fixed
        if target_budget <= 0:
            return [], []

        lists = [list(target_map[rid]) for rid in selected_ids]
        counts = [0] * len(lists)
        remaining = int(target_budget)

        # Round-robin by target rank. Because each list is importance sorted,
        # this preserves high-importance-first while preventing the first regulon
        # from monopolizing the decoder context.
        rank = 0
        while remaining > 0:
            progressed = False
            for i, targets in enumerate(lists):
                if rank < len(targets) and remaining > 0:
                    counts[i] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
            rank += 1

        blocks: List[Dict[str, Any]] = []
        kept_ids: List[int] = []
        for rid, targets, count in zip(selected_ids, lists, counts):
            if count <= 0:
                continue
            kept_ids.append(int(rid))
            blocks.append(
                {
                    "regulon_id": int(rid),
                    "tf_token_id": int(self.regulon_lookup[rid]["tf_token_id"]),
                    "target_token_ids": [int(x) for x in targets[:count]],
                }
            )
        return kept_ids, blocks


    # -----------------------------------------------------------------
    # Task availability / Decoder IO
    # -----------------------------------------------------------------

    @staticmethod
    def _value_is_available(value: Any) -> bool:
        if value is None:
            return False
        value = parse_json_if_needed(value)
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) > 0
        return True

    def _annotation_available(self, example: Dict[str, Any], task_cfg: Dict[str, Any]) -> bool:
        field = str(task_cfg.get("field", "natural_language_annotation"))
        return field in example and self._value_is_available(example[field])

    def _choose_weighted_task(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(tasks) == 1:
            return tasks[0]
        weights = [float(t.get("sampling_weight", 1.0)) for t in tasks]
        if sum(weights) <= 0:
            raise ValueError("sampling_weight 之和必须 > 0。")
        return random.choices(tasks, weights=weights, k=1)[0]

    def _build_decoder_ios(
        self,
        example: Dict[str, Any],
        active_ids: Sequence[int],
        raw_expressed_gene_ids: Sequence[int],
        cell_key: str,
    ) -> List[Dict[str, Any]]:
        expressed_set = set(int(x) for x in raw_expressed_gene_ids)
        eligible_ids, target_map = self._eligible_active_regulons(active_ids, expressed_set)
        selected_ids = self._select_regulon_ids(eligible_ids, cell_key)

        regulon_io: Optional[Dict[str, Any]] = None
        if selected_ids:
            if self.dynamic_target_budget:
                selected_ids, blocks = self._allocate_dynamic_target_budget(selected_ids, target_map)
            else:
                blocks = [
                    {
                        "regulon_id": rid,
                        "tf_token_id": int(self.regulon_lookup[rid]["tf_token_id"]),
                        "target_token_ids": target_map[rid],
                    }
                    for rid in selected_ids
                ]

            if blocks:
                query_tf_ids = [int(b["tf_token_id"]) for b in blocks]
                regulon_io = self.tokenizer.build_multi_regulon_decoder_io(
                    regulon_blocks=blocks,
                    query_tf_ids=query_tf_ids,
                    max_length=self.max_decoder_length,
                )
                regulon_io.update(
                    {
                        "decoder_task_id": REGULON_TASK_ID,
                        "selected_regulon_ids": [int(x) for x in selected_ids],
                        "query_tf_ids": query_tf_ids,
                        "regulon_blocks": blocks,
                    }
                )

        annotation_io: Optional[Dict[str, Any]] = None
        for task_cfg in self.decoder_tasks:
            if not task_cfg.get("enabled", True):
                continue
            if self._get_task_id(task_cfg) != ANNOTATION_TASK_ID:
                continue
            if not self._annotation_available(example, task_cfg):
                continue
            field = str(task_cfg.get("field", "natural_language_annotation"))
            annotation_io = self.tokenizer.build_generic_task_decoder_io(
                task_name="annotation",
                content=parse_json_if_needed(example[field]),
                target_type="plain_text",
                max_length=self.max_decoder_length,
            )
            annotation_io.update(
                {
                    "decoder_task_id": ANNOTATION_TASK_ID,
                    "selected_regulon_ids": [],
                    "query_tf_ids": [],
                    "regulon_blocks": [],
                }
            )
            break

        available: List[Dict[str, Any]] = []
        # Preserve configured task order where possible.
        by_id = {
            REGULON_TASK_ID: regulon_io,
            ANNOTATION_TASK_ID: annotation_io,
        }
        for task_cfg in self.decoder_tasks:
            if not task_cfg.get("enabled", True):
                continue
            tid = self._get_task_id(task_cfg)
            if by_id.get(tid) is not None:
                item = dict(by_id[tid])
                item["_sampling_weight"] = float(task_cfg.get("sampling_weight", 1.0))
                available.append(item)

        if not available:
            return []
        if self.task_sampling_mode == "all":
            return available
        chosen = self._choose_weighted_task(available)
        return [chosen]

    # -----------------------------------------------------------------
    # One sample
    # -----------------------------------------------------------------

    def _prepare_one(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        cell_key = self._stable_cell_key(example)
        mapped_genes, mapped_exprs = self._extract_raw_mapped_gene_expr(example)
        raw_expressed_set = self._raw_expressed_gene_set(mapped_genes, mapped_exprs)
        raw_expressed_gene_ids = sorted(raw_expressed_set)

        genes, exprs = self._prepare_gene_expr_from_mapped(
            mapped_genes,
            mapped_exprs,
            cell_key=cell_key,
        )
        encoder_pack = self._apply_encoder_mask(genes, exprs, cell_key=cell_key)

        active_ids = self._parse_active_regulon_ids(example)
        decoder_ios = self._build_decoder_ios(
            example=example,
            active_ids=active_ids,
            raw_expressed_gene_ids=raw_expressed_gene_ids,
            cell_key=cell_key,
        )
        if not decoder_ios:
            return []

        sample_weight = 1.0
        if self.use_loss_weight and self.loss_weight_field and self.loss_weight_field in example:
            try:
                sample_weight = float(example[self.loss_weight_field])
            except Exception:
                sample_weight = 1.0

        items: List[Dict[str, Any]] = []
        for decoder_io in decoder_ios:
            item = {
                **encoder_pack,
                **decoder_io,
                "sample_weight": float(sample_weight),
                "decoder_task_id": int(decoder_io["decoder_task_id"]),
                "selected_regulon_ids": [int(x) for x in decoder_io.get("selected_regulon_ids", [])],
                "query_tf_ids": [int(x) for x in decoder_io.get("query_tf_ids", [])],
                "regulon_blocks": decoder_io.get("regulon_blocks", []),
                "cell_key": cell_key,
            }
            items.append(item)
        return items

    # -----------------------------------------------------------------
    # Batch padding
    # -----------------------------------------------------------------

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for example in examples:
            items.extend(self._prepare_one(dict(example)))
        if not items:
            raise ValueError("当前 batch 没有生成任何训练样本。")

        batch_size = len(items)
        max_enc_len = max(item["input_gene_ids"].shape[0] for item in items)
        max_dec_len = max(len(item["decoder_input_ids"]) for item in items)

        encoder_input_gene_ids = torch.full((batch_size, max_enc_len), self.pad_token_id, dtype=torch.long)
        encoder_input_values = torch.full((batch_size, max_enc_len), self.pad_value, dtype=torch.float)
        encoder_target_gene_ids = torch.full((batch_size, max_enc_len), self.pad_token_id, dtype=torch.long)
        encoder_target_values = torch.full((batch_size, max_enc_len), self.pad_value, dtype=torch.float)
        gene_masks = torch.zeros((batch_size, max_enc_len), dtype=torch.bool)
        expr_masks = torch.zeros((batch_size, max_enc_len), dtype=torch.bool)
        encoder_key_padding_mask = torch.ones((batch_size, max_enc_len), dtype=torch.bool)

        decoder_input_ids = torch.full((batch_size, max_dec_len), self.tokenizer.pad_token_id, dtype=torch.long)
        decoder_labels = torch.full((batch_size, max_dec_len), -100, dtype=torch.long)
        decoder_attention_mask = torch.zeros((batch_size, max_dec_len), dtype=torch.long)
        decoder_task_ids = torch.full((batch_size,), -1, dtype=torch.long)
        sample_weights = torch.ones((batch_size,), dtype=torch.float)

        selected_regulon_ids_list: List[List[int]] = []
        query_tf_ids_list: List[List[int]] = []
        regulon_blocks_list: List[List[Dict[str, Any]]] = []
        cell_keys: List[str] = []

        for i, item in enumerate(items):
            enc_len = item["input_gene_ids"].shape[0]
            dec_len = len(item["decoder_input_ids"])

            encoder_input_gene_ids[i, :enc_len] = item["input_gene_ids"]
            encoder_input_values[i, :enc_len] = item["input_values"]
            encoder_target_gene_ids[i, :enc_len] = item["target_gene_ids"]
            encoder_target_values[i, :enc_len] = item["target_values"]
            gene_masks[i, :enc_len] = item["gene_masks"]
            expr_masks[i, :enc_len] = item["expr_masks"]
            encoder_key_padding_mask[i, :enc_len] = False

            decoder_input_ids[i, :dec_len] = torch.tensor(item["decoder_input_ids"], dtype=torch.long)
            decoder_labels[i, :dec_len] = torch.tensor(item["decoder_labels"], dtype=torch.long)
            decoder_attention_mask[i, :dec_len] = torch.tensor(item["decoder_attention_mask"], dtype=torch.long)
            decoder_task_ids[i] = int(item["decoder_task_id"])
            sample_weights[i] = float(item["sample_weight"])

            selected_regulon_ids_list.append(list(item["selected_regulon_ids"]))
            query_tf_ids_list.append(list(item["query_tf_ids"]))
            regulon_blocks_list.append(list(item["regulon_blocks"]))
            cell_keys.append(str(item["cell_key"]))

        return {
            "encoder_input_gene_ids": encoder_input_gene_ids,
            "encoder_input_values": encoder_input_values,
            "encoder_target_gene_ids": encoder_target_gene_ids,
            "encoder_target_values": encoder_target_values,
            "gene_masks": gene_masks,
            "expr_masks": expr_masks,
            "encoder_key_padding_mask": encoder_key_padding_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_labels": decoder_labels,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_task_ids": decoder_task_ids,
            "sample_weights": sample_weights,
            # Python metadata retained for debugging and inspection.
            "selected_regulon_ids": selected_regulon_ids_list,
            "query_tf_ids": query_tf_ids_list,
            "regulon_blocks_list": regulon_blocks_list,
            "cell_keys": cell_keys,
        }


# =====================================================================
# Streaming DataLoader
# =====================================================================

def build_streaming_dataloader(
    local: PathLike,
    tokenizer: GlobalGeneTextTokenizer,
    batch_size: int = 16,
    max_encoder_length: int = 2049,
    max_decoder_length: int = 1025,
    mlm_probability: float = 0.10,
    mask_gene_input: bool = False,
    mask_expr_input: bool = True,
    num_bins: int = 51,
    sampling: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    decoder_tasks: Optional[List[Dict[str, Any]]] = None,
    genes_field: str = "genes",
    expressions_field: str = "expressions",
    loss_weight_field: Optional[str] = "loss_weight",
    keep_first_n_tokens: int = 1,
    pad_value: float = -2.0,
    cls_value: float = -1.0,
    mask_value: float = -3.0,
    use_loss_weight: bool = False,
    task_sampling_mode: str = "all",
    regulon_target_path: Optional[PathLike] = None,
    active_regulon_field: str = "active_regulon_ids",
    cell_id_field: str = "cell_id",
    n_regulons: int = 530,
    regulon_num_queries: int = 3,
    regulon_sampling_mode: str = "cyclic_without_replacement",
    regulon_seed: int = 42,
    validation_regulon_seed: int = 2026,
    cell_specific_targets: bool = True,
    expressed_gene_threshold: float = 0.0,
    dynamic_target_budget: bool = True,
    is_training: bool = True,
    fixed_eval_encoder_mask: bool = True,
) -> DataLoader:
    dataset = StreamingDataset(
        local=str(local),
        shuffle=bool(shuffle),
        batch_size=int(batch_size),
        allow_unsafe_types=True,
    )

    collator = Stage2Collator(
        tokenizer=tokenizer,
        max_encoder_length=max_encoder_length,
        max_decoder_length=max_decoder_length,
        mlm_probability=mlm_probability,
        num_bins=num_bins,
        sampling=sampling,
        keep_first_n_tokens=keep_first_n_tokens,
        pad_value=pad_value,
        cls_value=cls_value,
        mask_value=mask_value,
        mask_gene_input=mask_gene_input,
        mask_expr_input=mask_expr_input,
        decoder_tasks=decoder_tasks,
        genes_field=genes_field,
        expressions_field=expressions_field,
        loss_weight_field=loss_weight_field,
        use_loss_weight=use_loss_weight,
        task_sampling_mode=task_sampling_mode,
        regulon_target_path=regulon_target_path,
        active_regulon_field=active_regulon_field,
        cell_id_field=cell_id_field,
        n_regulons=n_regulons,
        regulon_num_queries=regulon_num_queries,
        regulon_sampling_mode=regulon_sampling_mode,
        regulon_seed=regulon_seed,
        validation_regulon_seed=validation_regulon_seed,
        cell_specific_targets=cell_specific_targets,
        expressed_gene_threshold=expressed_gene_threshold,
        dynamic_target_budget=dynamic_target_budget,
        is_training=is_training,
        fixed_eval_encoder_mask=fixed_eval_encoder_mask,
    )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=collator,
        # Keep false so collator.set_epoch(epoch) is propagated when workers
        # are recreated at each new DataLoader iteration / epoch.
        persistent_workers=False,
    )
