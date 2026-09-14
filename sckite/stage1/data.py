#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import zlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from streaming import StreamingDataset
except Exception as exc:
    raise ImportError(
        "无法导入 streaming.StreamingDataset，请先安装 "
        "mosaicml-streaming / streaming。"
    ) from exc

from sckite.stage1.tokenizer import GlobalGeneTextTokenizer


PathLike = Union[str, Path]


def left_binning(
    values: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    if values.numel() == 0:
        return values

    out = torch.zeros_like(
        values,
        dtype=torch.long,
    )
    pos_mask = values > 0

    if pos_mask.sum() == 0:
        return out

    pos_values = values[pos_mask].float()
    if pos_values.numel() == 1:
        out[pos_mask] = 1
        return out

    q = torch.linspace(
        0.0,
        1.0,
        steps=int(num_bins),
        device=values.device,
    )
    edges = torch.quantile(
        pos_values,
        q,
    )
    inner_edges = edges[1:-1].contiguous()
    out[pos_mask] = (
        torch.bucketize(
            pos_values,
            inner_edges,
            right=False,
        ).long()
        + 1
    )
    return out


def sample_or_truncate(
    genes: torch.Tensor,
    exprs: torch.Tensor,
    max_length: int,
    keep_first_n_tokens: int = 1,
    sampling: bool = True,
    generator: Optional[
        torch.Generator
    ] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if genes.ndim != 1 or exprs.ndim != 1:
        raise ValueError(
            "genes 和 exprs 必须是一维张量。"
        )
    if genes.shape[0] != exprs.shape[0]:
        raise ValueError(
            "genes 与 exprs 长度必须一致。"
        )

    if genes.shape[0] <= int(max_length):
        return genes, exprs

    prefix_genes = genes[
        :keep_first_n_tokens
    ]
    prefix_exprs = exprs[
        :keep_first_n_tokens
    ]
    body_genes = genes[
        keep_first_n_tokens:
    ]
    body_exprs = exprs[
        keep_first_n_tokens:
    ]

    need = (
        int(max_length)
        - int(keep_first_n_tokens)
    )
    if need <= 0:
        return (
            prefix_genes[:max_length],
            prefix_exprs[:max_length],
        )

    if sampling:
        idx = torch.randperm(
            body_genes.shape[0],
            generator=generator,
        )[:need]
        idx, _ = torch.sort(idx)
    else:
        idx = torch.arange(need)

    return (
        torch.cat(
            [prefix_genes, body_genes[idx]],
            dim=0,
        ),
        torch.cat(
            [prefix_exprs, body_exprs[idx]],
            dim=0,
        ),
    )


def parse_json_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if (
            (
                text.startswith("[")
                and text.endswith("]")
            )
            or (
                text.startswith("{")
                and text.endswith("}")
            )
        ):
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _stable_crc32(text: str) -> int:
    return int(
        zlib.crc32(
            str(text).encode("utf-8")
        )
        & 0xFFFFFFFF
    )


class EncoderOnlyCollator:
    """
    One raw cell produces exactly one encoder record.

    No decoder task expansion, Regulon lookup, active-Regulon label,
    annotation target, or test-specific metadata is constructed.
    """

    def __init__(
        self,
        tokenizer: GlobalGeneTextTokenizer,
        max_encoder_length: int = 2049,
        mlm_probability: float = 0.10,
        num_bins: int = 51,
        sampling: bool = False,
        keep_first_n_tokens: int = 1,
        pad_value: float = -2.0,
        cls_value: float = -1.0,
        mask_value: float = -3.0,
        mask_gene_input: bool = False,
        mask_expr_input: bool = True,
        genes_field: str = "genes",
        expressions_field: str = "expressions",
        cell_id_field: str = "cell_id",
        is_training: bool = True,
        fixed_eval_encoder_mask: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_encoder_length = int(
            max_encoder_length
        )
        self.mlm_probability = float(
            mlm_probability
        )
        self.num_bins = int(num_bins)
        self.sampling = bool(sampling)
        self.keep_first_n_tokens = int(
            keep_first_n_tokens
        )
        self.pad_value = float(pad_value)
        self.cls_value = float(cls_value)
        self.mask_value = float(mask_value)
        self.mask_gene_input = bool(
            mask_gene_input
        )
        self.mask_expr_input = bool(
            mask_expr_input
        )
        self.genes_field = str(genes_field)
        self.expressions_field = str(
            expressions_field
        )
        self.cell_id_field = str(
            cell_id_field
        )
        self.is_training = bool(is_training)
        self.fixed_eval_encoder_mask = bool(
            fixed_eval_encoder_mask
        )

        self.pad_token_id = int(
            self.tokenizer.pad_token_id
        )
        self.cls_token_id = int(
            self.tokenizer.cls_token_id
        )
        self.mask_gene_token_id = int(
            self.tokenizer.mask_gene_token_id
        )

    def _stable_cell_key(
        self,
        example: Dict[str, Any],
    ) -> str:
        for key in [
            self.cell_id_field,
            "cell_id",
            "barcode",
            "obs_names",
            "index",
        ]:
            if key and key in example:
                value = example[key]
                if (
                    value is not None
                    and str(value).strip()
                ):
                    return str(value)

        genes = parse_json_if_needed(
            example.get(
                self.genes_field,
                [],
            )
        )
        exprs = parse_json_if_needed(
            example.get(
                self.expressions_field,
                [],
            )
        )
        try:
            preview = [
                list(genes)[:64],
                [
                    float(x)
                    for x in list(exprs)[:64]
                ],
            ]
        except Exception:
            preview = [[], []]

        return (
            "fallback:"
            + str(
                _stable_crc32(
                    json.dumps(
                        preview,
                        default=str,
                    )
                )
            )
        )

    def _make_generator(
        self,
        key: str,
        salt: int,
    ) -> torch.Generator:
        seed = (
            _stable_crc32(
                f"{key}|{salt}"
            )
            + int(salt)
        ) % (2**31 - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _map_gene_value_to_global_id(
        self,
        gene_value: Any,
    ) -> Optional[int]:
        if gene_value is None:
            return None

        if torch.is_tensor(gene_value):
            gene_value = gene_value.item()

        if (
            hasattr(gene_value, "item")
            and not isinstance(
                gene_value,
                (str, bytes),
            )
        ):
            try:
                gene_value = gene_value.item()
            except Exception:
                pass

        if not isinstance(gene_value, bool):
            try:
                text = str(
                    gene_value
                ).strip()
                if (
                    isinstance(
                        gene_value,
                        (int, float),
                    )
                    or text.isdigit()
                ):
                    gid = int(gene_value)
                    if (
                        gid
                        in self.tokenizer
                        .global_id_to_token
                        and self.tokenizer
                        .global_id_is_gene(gid)
                    ):
                        return gid
            except Exception:
                pass

        return (
            self.tokenizer
            .gene_name_to_global_id(
                str(gene_value)
            )
        )

    def _extract_mapped_gene_expr(
        self,
        example: Dict[str, Any],
    ) -> Tuple[list[int], list[float]]:
        if self.genes_field not in example:
            raise KeyError(
                f"样本缺少 genes 字段 "
                f"{self.genes_field!r}。"
            )
        if (
            self.expressions_field
            not in example
        ):
            raise KeyError(
                "样本缺少 expressions 字段 "
                f"{self.expressions_field!r}。"
            )

        raw_genes = list(
            parse_json_if_needed(
                example[self.genes_field]
            )
        )
        raw_exprs = list(
            parse_json_if_needed(
                example[
                    self.expressions_field
                ]
            )
        )

        if len(raw_genes) != len(raw_exprs):
            raise ValueError(
                "genes 与 expressions 长度不一致："
                f"{len(raw_genes)} vs "
                f"{len(raw_exprs)}"
            )

        mapped_genes: list[int] = []
        mapped_exprs: list[float] = []

        for gene, expr in zip(
            raw_genes,
            raw_exprs,
        ):
            gid = (
                self
                ._map_gene_value_to_global_id(
                    gene
                )
            )
            if gid is None:
                continue
            mapped_genes.append(int(gid))
            mapped_exprs.append(float(expr))

        return mapped_genes, mapped_exprs

    def _prepare_encoder_sequence(
        self,
        mapped_genes: Sequence[int],
        mapped_exprs: Sequence[float],
        cell_key: str,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        genes = torch.as_tensor(
            list(mapped_genes),
            dtype=torch.long,
        )
        exprs = torch.as_tensor(
            list(mapped_exprs),
            dtype=torch.float,
        )

        genes = torch.cat(
            [
                torch.tensor(
                    [self.cls_token_id],
                    dtype=torch.long,
                ),
                genes,
            ],
            dim=0,
        )
        exprs = torch.cat(
            [
                torch.tensor(
                    [self.cls_value],
                    dtype=torch.float,
                ),
                exprs,
            ],
            dim=0,
        )

        generator = None
        if (
            self.sampling
            and not self.is_training
            and self.fixed_eval_encoder_mask
        ):
            generator = self._make_generator(
                cell_key,
                99173,
            )

        return sample_or_truncate(
            genes=genes,
            exprs=exprs,
            max_length=
                self.max_encoder_length,
            keep_first_n_tokens=
                self.keep_first_n_tokens,
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

        if (
            binned_values.shape[0]
            > self.keep_first_n_tokens
        ):
            binned_values[
                self.keep_first_n_tokens:
            ] = left_binning(
                binned_values[
                    self.keep_first_n_tokens:
                ],
                self.num_bins,
            ).float()

        target_values = binned_values.clone()
        input_gene_ids = genes.clone()
        input_values = binned_values.clone()

        valid_mask = torch.zeros_like(
            genes,
            dtype=torch.bool,
        )
        valid_mask[
            self.keep_first_n_tokens:
        ] = True
        valid_mask &= genes.ne(
            self.pad_token_id
        )

        if self.mlm_probability <= 0:
            rand_mask = torch.zeros_like(
                valid_mask,
                dtype=torch.bool,
            )
        else:
            generator = None
            if (
                not self.is_training
                and self.fixed_eval_encoder_mask
            ):
                generator = self._make_generator(
                    cell_key,
                    314159,
                )

            rand = torch.rand(
                genes.shape,
                dtype=torch.float,
                generator=generator,
            )
            rand_mask = (
                rand.lt(
                    self.mlm_probability
                )
                & valid_mask
            )

            if (
                valid_mask.any()
                and not rand_mask.any()
            ):
                valid_indices = torch.where(
                    valid_mask
                )[0]
                if generator is None:
                    index = torch.randint(
                        0,
                        valid_indices.numel(),
                        (1,),
                    ).item()
                else:
                    index = torch.randint(
                        0,
                        valid_indices.numel(),
                        (1,),
                        generator=generator,
                    ).item()
                rand_mask[
                    valid_indices[index]
                ] = True

        if self.mask_gene_input:
            input_gene_ids[
                rand_mask
            ] = self.mask_gene_token_id
            gene_masks = rand_mask.clone()
        else:
            gene_masks = torch.zeros_like(
                rand_mask,
                dtype=torch.bool,
            )

        if self.mask_expr_input:
            input_values[
                rand_mask
            ] = self.mask_value
            expr_masks = rand_mask.clone()
        else:
            expr_masks = torch.zeros_like(
                rand_mask,
                dtype=torch.bool,
            )

        return {
            "input_gene_ids":
                input_gene_ids,
            "input_values":
                input_values,
            "target_gene_ids":
                target_gene_ids,
            "target_values":
                target_values,
            "gene_masks":
                gene_masks,
            "expr_masks":
                expr_masks,
        }

    def _prepare_one(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        cell_key = self._stable_cell_key(
            example
        )
        mapped_genes, mapped_exprs = (
            self._extract_mapped_gene_expr(
                example
            )
        )
        genes, exprs = (
            self._prepare_encoder_sequence(
                mapped_genes,
                mapped_exprs,
                cell_key,
            )
        )
        encoder_pack = (
            self._apply_encoder_mask(
                genes,
                exprs,
                cell_key,
            )
        )
        encoder_pack["cell_key"] = cell_key
        return encoder_pack

    def __call__(
        self,
        examples: Sequence[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        items = [
            self._prepare_one(
                dict(example)
            )
            for example in examples
        ]

        if not items:
            raise ValueError(
                "当前 batch 没有样本。"
            )

        batch_size = len(items)
        max_enc_len = max(
            item["input_gene_ids"].shape[0]
            for item in items
        )

        encoder_input_gene_ids = torch.full(
            (batch_size, max_enc_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        encoder_input_values = torch.full(
            (batch_size, max_enc_len),
            self.pad_value,
            dtype=torch.float,
        )
        encoder_target_gene_ids = torch.full(
            (batch_size, max_enc_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        encoder_target_values = torch.full(
            (batch_size, max_enc_len),
            self.pad_value,
            dtype=torch.float,
        )
        gene_masks = torch.zeros(
            (batch_size, max_enc_len),
            dtype=torch.bool,
        )
        expr_masks = torch.zeros(
            (batch_size, max_enc_len),
            dtype=torch.bool,
        )
        encoder_key_padding_mask = torch.ones(
            (batch_size, max_enc_len),
            dtype=torch.bool,
        )
        cell_keys: list[str] = []

        for i, item in enumerate(items):
            length = item[
                "input_gene_ids"
            ].shape[0]

            encoder_input_gene_ids[
                i, :length
            ] = item["input_gene_ids"]
            encoder_input_values[
                i, :length
            ] = item["input_values"]
            encoder_target_gene_ids[
                i, :length
            ] = item["target_gene_ids"]
            encoder_target_values[
                i, :length
            ] = item["target_values"]
            gene_masks[
                i, :length
            ] = item["gene_masks"]
            expr_masks[
                i, :length
            ] = item["expr_masks"]
            encoder_key_padding_mask[
                i, :length
            ] = False
            cell_keys.append(
                str(item["cell_key"])
            )

        return {
            "encoder_input_gene_ids":
                encoder_input_gene_ids,
            "encoder_input_values":
                encoder_input_values,
            "encoder_target_gene_ids":
                encoder_target_gene_ids,
            "encoder_target_values":
                encoder_target_values,
            "gene_masks":
                gene_masks,
            "expr_masks":
                expr_masks,
            "encoder_key_padding_mask":
                encoder_key_padding_mask,
            "cell_keys":
                cell_keys,
        }


def build_streaming_dataloader(
    local: PathLike,
    tokenizer: GlobalGeneTextTokenizer,
    batch_size: int = 16,
    max_encoder_length: int = 2049,
    mlm_probability: float = 0.10,
    mask_gene_input: bool = False,
    mask_expr_input: bool = True,
    num_bins: int = 51,
    sampling: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    genes_field: str = "genes",
    expressions_field: str = "expressions",
    cell_id_field: str = "cell_id",
    keep_first_n_tokens: int = 1,
    pad_value: float = -2.0,
    cls_value: float = -1.0,
    mask_value: float = -3.0,
    is_training: bool = True,
    fixed_eval_encoder_mask: bool = True,
) -> DataLoader:
    dataset = StreamingDataset(
        local=str(local),
        shuffle=bool(shuffle),
        batch_size=int(batch_size),
        allow_unsafe_types=True,
    )

    collator = EncoderOnlyCollator(
        tokenizer=tokenizer,
        max_encoder_length=
            max_encoder_length,
        mlm_probability=mlm_probability,
        num_bins=num_bins,
        sampling=sampling,
        keep_first_n_tokens=
            keep_first_n_tokens,
        pad_value=pad_value,
        cls_value=cls_value,
        mask_value=mask_value,
        mask_gene_input=mask_gene_input,
        mask_expr_input=mask_expr_input,
        genes_field=genes_field,
        expressions_field=
            expressions_field,
        cell_id_field=cell_id_field,
        is_training=is_training,
        fixed_eval_encoder_mask=
            fixed_eval_encoder_mask,
    )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=collator,
        persistent_workers=False,
    )
