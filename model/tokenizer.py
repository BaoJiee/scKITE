#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:
    from transformers import BertTokenizer, BertTokenizerFast
except Exception as e:
    raise ImportError("无法导入 transformers，请先安装：pip install transformers") from e


PathLike = Union[str, Path]


class GlobalGeneTextTokenizer:
    """
    Stage2 统一 global-vocab tokenizer。

    正式支持：
    1. Annotation / plain_text
    2. Regulon / multi_regulon_compact

    Multi-Regulon 结构：
      <task> regulon
      TF1 <gene_sep> TF2 <gene_sep> TF3
      <startofanswer>
      TF1 <arrow> T1 T2 ...
      <regulon_sep>
      TF2 <arrow> T1 T2 ...
      <regulon_sep>
      TF3 <arrow> T1 T2 ...
      <eos>

    为兼容旧代码，仍保留 gene_pairs / regulon_edges 接口，但正式训练应使用
    build_multi_regulon_decoder_io()。
    """

    def __init__(
        self,
        text_tokenizer: Union[BertTokenizer, BertTokenizerFast],
        tokenizer_name_or_path: str,
        global_vocab: Dict[str, int],
        global_vocab_meta: Dict[str, Any],
        gene_table_records: List[Dict[str, Any]],
        max_length: int = 128,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.tokenizer_name_or_path = str(tokenizer_name_or_path)
        self.max_length = int(max_length)

        self.global_vocab = {str(k): int(v) for k, v in global_vocab.items()}
        self.global_id_to_token = {int(v): str(k) for k, v in self.global_vocab.items()}
        if len(self.global_id_to_token) != len(self.global_vocab):
            raise ValueError("global_vocab 中存在重复 global id，请检查 global_vocab.json。")

        self.meta = dict(global_vocab_meta)
        self.gene_table_records = list(gene_table_records)

        self.pad_token = str(self.meta.get("pad_token", "<pad>"))
        self.cls_token = str(self.meta.get("cls_token", "<cls>"))
        self.bos_token = str(self.meta.get("bos_token", "<bos>"))
        self.eos_token = str(self.meta.get("eos_token", "<eos>"))
        self.unk_token = str(self.meta.get("unk_token", "<unk>"))
        self.mask_gene_token = str(self.meta.get("mask_gene_token", "<mask_gene>"))
        self.task_token = str(self.meta.get("task_token", "<task>"))
        self.start_answer_token = str(self.meta.get("start_answer_token", "<startofanswer>"))
        self.end_answer_token = str(self.meta.get("end_answer_token", "<endofanswer>"))
        self.gene_sep_token = str(self.meta.get("gene_sep_token", "<gene_sep>"))
        self.regulon_sep_token = str(self.meta.get("regulon_sep_token", "<regulon_sep>"))
        self.arrow_token = str(self.meta.get("arrow_token", "<arrow>"))

        required_specials = [
            self.pad_token,
            self.cls_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.mask_gene_token,
            self.task_token,
            self.start_answer_token,
            self.end_answer_token,
            self.gene_sep_token,
            self.regulon_sep_token,
            self.arrow_token,
        ]
        missing = [tok for tok in required_specials if tok not in self.global_vocab]
        if missing:
            raise KeyError(f"global_vocab 缺少必要特殊 token：{missing}")

        self.pad_token_id = int(self.global_vocab[self.pad_token])
        self.cls_token_id = int(self.global_vocab[self.cls_token])
        self.bos_token_id = int(self.global_vocab[self.bos_token])
        self.eos_token_id = int(self.global_vocab[self.eos_token])
        self.unk_token_id = int(self.global_vocab[self.unk_token])
        self.mask_gene_token_id = int(self.global_vocab[self.mask_gene_token])
        self.task_token_id = int(self.global_vocab[self.task_token])
        self.start_answer_token_id = int(self.global_vocab[self.start_answer_token])
        self.end_answer_token_id = int(self.global_vocab[self.end_answer_token])
        self.gene_sep_token_id = int(self.global_vocab[self.gene_sep_token])
        self.regulon_sep_token_id = int(self.global_vocab[self.regulon_sep_token])
        self.arrow_token_id = int(self.global_vocab[self.arrow_token])

        self.gene_token_prefix = str(self.meta.get("gene_token_prefix", "<gene:"))
        self.text_token_prefix = str(self.meta.get("text_token_prefix", "<txt:"))
        self.token_suffix = str(self.meta.get("token_suffix", ">"))
        self.bert_special_to_global = dict(self.meta.get("bert_special_to_global", {}))

        self.ensembl_to_global_id: Dict[str, int] = {}
        self.global_id_to_ensembl: Dict[int, str] = {}
        self.global_id_to_gene_symbol: Dict[int, str] = {}
        self.symbol_to_ids: Dict[str, List[int]] = defaultdict(list)
        self._build_gene_maps()

    # ------------------------------------------------------------------
    # Loading / vocabulary
    # ------------------------------------------------------------------

    def _build_gene_maps(self) -> None:
        for rec in self.gene_table_records:
            if not isinstance(rec, dict) or "global_id" not in rec:
                continue
            global_id = int(rec["global_id"])
            ensembl_id = re.sub(r"\.\d+$", "", str(rec.get("ensembl_id", "")).strip())
            gene_symbol = str(rec.get("gene_symbol", ensembl_id)).strip()

            if ensembl_id:
                self.ensembl_to_global_id[ensembl_id] = global_id
                self.global_id_to_ensembl[global_id] = ensembl_id
                self.symbol_to_ids[ensembl_id].append(global_id)

            if gene_symbol and gene_symbol.lower() != "nan":
                self.global_id_to_gene_symbol[global_id] = gene_symbol
                self.symbol_to_ids[gene_symbol].append(global_id)

        self.symbol_to_primary_id: Dict[str, int] = {
            str(symbol): sorted(set(ids))[0]
            for symbol, ids in self.symbol_to_ids.items()
        }
        self.ambiguous_symbols: Dict[str, List[int]] = {
            str(symbol): sorted(set(ids))
            for symbol, ids in self.symbol_to_ids.items()
            if len(set(ids)) > 1
        }

    @staticmethod
    def read_json_or_jsonl(path: PathLike) -> Any:
        path = Path(path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"文件为空：{path}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            records = []
            for line_no, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"第 {line_no} 行不是合法 JSON：{line[:120]}") from e
            return records

    @staticmethod
    def load_bert_tokenizer(
        tokenizer_name_or_path: PathLike,
        use_fast: bool = True,
        do_lower_case: bool = False,
    ):
        tokenizer_name_or_path = str(tokenizer_name_or_path)
        path_obj = Path(tokenizer_name_or_path)

        if path_obj.is_file() and path_obj.name == "vocab.txt":
            if use_fast:
                try:
                    return BertTokenizerFast(
                        vocab_file=tokenizer_name_or_path,
                        do_lower_case=bool(do_lower_case),
                    )
                except Exception:
                    pass
            return BertTokenizer(
                vocab_file=tokenizer_name_or_path,
                do_lower_case=bool(do_lower_case),
            )

        if use_fast:
            try:
                return BertTokenizerFast.from_pretrained(
                    tokenizer_name_or_path,
                    do_lower_case=bool(do_lower_case),
                )
            except Exception:
                pass

        return BertTokenizer.from_pretrained(
            tokenizer_name_or_path,
            do_lower_case=bool(do_lower_case),
        )

    @classmethod
    def from_files(
        cls,
        text_tokenizer_path: PathLike,
        global_vocab_path: PathLike,
        global_vocab_meta_path: PathLike,
        gene_table_path: PathLike,
        max_length: int = 128,
        use_fast: bool = True,
        do_lower_case: bool = False,
    ) -> "GlobalGeneTextTokenizer":
        return cls(
            text_tokenizer=cls.load_bert_tokenizer(
                text_tokenizer_path,
                use_fast=use_fast,
                do_lower_case=do_lower_case,
            ),
            tokenizer_name_or_path=str(text_tokenizer_path),
            global_vocab=cls.read_json_or_jsonl(global_vocab_path),
            global_vocab_meta=cls.read_json_or_jsonl(global_vocab_meta_path),
            gene_table_records=cls.read_json_or_jsonl(gene_table_path),
            max_length=max_length,
        )

    @property
    def vocab_size(self) -> int:
        return int(
            self.meta.get(
                "vocab_size_for_embedding",
                max(self.global_vocab.values()) + 1,
            )
        )

    # ------------------------------------------------------------------
    # Gene / text mapping
    # ------------------------------------------------------------------

    def normalize_text(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = text.replace("\u00A0", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
        return " ".join(text.split())

    def make_text_token(self, bert_token: str) -> str:
        return f"{self.text_token_prefix}{str(bert_token)}{self.token_suffix}"

    def global_id_is_gene(self, global_id: int) -> bool:
        token = self.global_id_to_token.get(int(global_id), "")
        return token.startswith(self.gene_token_prefix) and token.endswith(self.token_suffix)

    def global_id_is_text(self, global_id: int) -> bool:
        token = self.global_id_to_token.get(int(global_id), "")
        return token.startswith(self.text_token_prefix) and token.endswith(self.token_suffix)

    def gene_name_to_global_id(self, gene_name: str) -> Optional[int]:
        """将 gene symbol / Ensembl / <gene:...> 映射到 global gene ID。"""
        if gene_name is None:
            return None
        text = str(gene_name).strip()
        if not text:
            return None

        # 已经是 global id 的字符串。
        if text.isdigit():
            gid = int(text)
            if gid in self.global_id_to_token and self.global_id_is_gene(gid):
                return gid

        # 直接 global token。
        if text in self.global_vocab:
            gid = int(self.global_vocab[text])
            if self.global_id_is_gene(gid):
                return gid

        # <gene:SYMBOL> / <gene:ENSEMBL>
        if text.startswith(self.gene_token_prefix) and text.endswith(self.token_suffix):
            inner = text[len(self.gene_token_prefix):-len(self.token_suffix)]
            text = inner

        no_version = re.sub(r"\.\d+$", "", text)
        if no_version in self.ensembl_to_global_id:
            return int(self.ensembl_to_global_id[no_version])
        if text in self.symbol_to_primary_id:
            return int(self.symbol_to_primary_id[text])
        if no_version in self.symbol_to_primary_id:
            return int(self.symbol_to_primary_id[no_version])

        for candidate in (f"{self.gene_token_prefix}{text}{self.token_suffix}",
                          f"{self.gene_token_prefix}{no_version}{self.token_suffix}"):
            if candidate in self.global_vocab:
                gid = int(self.global_vocab[candidate])
                if self.global_id_is_gene(gid):
                    return gid
        return None

    def bert_token_to_global_id(self, bert_token: str) -> int:
        bert_token = str(bert_token)
        if bert_token in self.bert_special_to_global:
            global_special = self.bert_special_to_global[bert_token]
            return int(self.global_vocab.get(global_special, self.unk_token_id))
        return int(self.global_vocab.get(self.make_text_token(bert_token), self.unk_token_id))

    def encode_plain_text_to_global_ids(self, text: str) -> List[int]:
        text = self.normalize_text(text)
        if not text:
            return []
        local_ids = self.text_tokenizer.encode(text, add_special_tokens=False, truncation=False)
        local_tokens = self.text_tokenizer.convert_ids_to_tokens(local_ids)
        return [int(self.bert_token_to_global_id(token)) for token in local_tokens]

    def encode_task_name_to_global_ids(self, task_name: str) -> List[int]:
        return self.encode_plain_text_to_global_ids(
            ("" if task_name is None else str(task_name)).replace("_", " ")
        )

    def _validate_global_gene_id(self, gene_id: int, role: str) -> int:
        gene_id = int(gene_id)
        if gene_id < 0 or gene_id >= self.vocab_size:
            raise ValueError(f"{role} gene id={gene_id} 超出 global vocab 范围 [0, {self.vocab_size})。")
        if gene_id not in self.global_id_to_token:
            raise ValueError(f"{role} gene id={gene_id} 不存在于 global_vocab。")
        if not self.global_id_is_gene(gene_id):
            token = self.global_id_to_token.get(gene_id, "<unknown>")
            raise ValueError(f"{role} id={gene_id} 对应 token={token!r}，不是 global gene token。")
        return gene_id

    # ------------------------------------------------------------------
    # New Multi-Regulon compact encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_regulon_block(block: Any) -> Tuple[Optional[int], int, List[int]]:
        """返回 (regulon_id, tf_token_id, target_token_ids)。"""
        if isinstance(block, dict):
            rid = block.get("regulon_id")
            tf = block.get("tf_token_id")
            targets = block.get("target_token_ids", [])
        elif isinstance(block, (list, tuple)) and len(block) == 2:
            rid = None
            tf, targets = block
        else:
            raise TypeError(
                "regulon block 必须是 {'tf_token_id':..., 'target_token_ids':...} "
                "或 (tf_token_id, target_token_ids)。"
            )
        if tf is None:
            raise ValueError(f"regulon block 缺少 tf_token_id：{block}")
        return (
            None if rid is None else int(rid),
            int(tf),
            [int(x) for x in list(targets)],
        )

    def encode_regulon_query_ids(self, query_tf_ids: Sequence[int]) -> List[int]:
        query_tf_ids = [self._validate_global_gene_id(int(x), "Query TF") for x in query_tf_ids]
        if not query_tf_ids:
            raise ValueError("query_tf_ids 不能为空。")
        if len(set(query_tf_ids)) != len(query_tf_ids):
            raise ValueError(f"query_tf_ids 中存在重复 TF：{query_tf_ids}")

        out: List[int] = []
        for i, tf_id in enumerate(query_tf_ids):
            if i > 0:
                out.append(int(self.gene_sep_token_id))
            out.append(int(tf_id))
        return out

    def encode_multi_regulon_compact_body(
        self,
        regulon_blocks: Sequence[Any],
    ) -> List[int]:
        """
        编码为：
          TF1 <arrow> T1 T2 ... <regulon_sep>
          TF2 <arrow> T1 T2 ...
        """
        if not regulon_blocks:
            raise ValueError("regulon_blocks 不能为空。")

        body: List[int] = []
        for block_idx, block in enumerate(regulon_blocks):
            _, tf_id, target_ids = self._normalize_regulon_block(block)
            tf_id = self._validate_global_gene_id(tf_id, "TF")
            target_ids = [self._validate_global_gene_id(t, "Target") for t in target_ids]
            if not target_ids:
                raise ValueError(f"TF={tf_id} 的 compact regulon block 没有 target。")

            if block_idx > 0:
                body.append(int(self.regulon_sep_token_id))
            body.extend([int(tf_id), int(self.arrow_token_id)])
            body.extend(int(t) for t in target_ids)
        return body

    def multi_regulon_fixed_token_count(
        self,
        query_tf_ids: Sequence[int],
        task_name: str = "regulon",
    ) -> int:
        """完整 target 中除 target gene tokens 外的固定 token 数。"""
        k = len(list(query_tf_ids))
        if k <= 0:
            raise ValueError("query_tf_ids 不能为空。")
        task_name_len = len(self.encode_task_name_to_global_ids(task_name))
        query_len = k + max(k - 1, 0)             # TF + gene_sep
        answer_structure_len = 2 * k + max(k - 1, 0)  # TF + arrow + regulon_sep
        return int(
            1                       # <task>
            + task_name_len
            + query_len
            + 1                     # <startofanswer>
            + answer_structure_len
            + 1                     # <eos>
        )

    def estimate_multi_regulon_total_length(
        self,
        regulon_blocks: Sequence[Any],
        query_tf_ids: Optional[Sequence[int]] = None,
        task_name: str = "regulon",
    ) -> int:
        blocks = [self._normalize_regulon_block(b) for b in regulon_blocks]
        if query_tf_ids is None:
            query_tf_ids = [tf for _, tf, _ in blocks]
        n_targets = sum(len(targets) for _, _, targets in blocks)
        return self.multi_regulon_fixed_token_count(query_tf_ids, task_name=task_name) + n_targets

    def build_multi_regulon_target_ids(
        self,
        regulon_blocks: Sequence[Any],
        query_tf_ids: Optional[Sequence[int]] = None,
        task_name: str = "regulon",
    ) -> Dict[str, Any]:
        blocks = [self._normalize_regulon_block(b) for b in regulon_blocks]
        if not blocks:
            raise ValueError("regulon_blocks 不能为空。")

        block_tfs = [tf for _, tf, _ in blocks]
        if query_tf_ids is None:
            query_tf_ids = block_tfs
        query_tf_ids = [int(x) for x in query_tf_ids]
        if query_tf_ids != block_tfs:
            raise ValueError(
                "query_tf_ids 顺序必须与 regulon_blocks 的 TF 顺序完全一致。"
                f" query={query_tf_ids}, blocks={block_tfs}"
            )

        query_ids = self.encode_regulon_query_ids(query_tf_ids)
        body_ids = self.encode_multi_regulon_compact_body(regulon_blocks)
        task_name_ids = self.encode_task_name_to_global_ids(task_name)

        target_ids = (
            [int(self.task_token_id)]
            + task_name_ids
            + query_ids
            + [int(self.start_answer_token_id)]
            + body_ids
            + [int(self.eos_token_id)]
        )
        return {
            "target_ids": target_ids,
            "query_tf_ids": query_tf_ids,
            "regulon_blocks": [
                {
                    "regulon_id": rid,
                    "tf_token_id": tf,
                    "target_token_ids": targets,
                }
                for rid, tf, targets in blocks
            ],
        }

    def build_multi_regulon_decoder_io(
        self,
        regulon_blocks: Sequence[Any],
        query_tf_ids: Optional[Sequence[int]] = None,
        task_name: str = "regulon",
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        target_pack = self.build_multi_regulon_target_ids(
            regulon_blocks=regulon_blocks,
            query_tf_ids=query_tf_ids,
            task_name=task_name,
        )
        decoder_io = self.make_decoder_input_and_labels(
            target_pack["target_ids"],
            max_length=max_length,
            allow_truncation=False,
        )
        decoder_io.update({k: v for k, v in target_pack.items() if k != "target_ids"})
        decoder_io["task_name"] = str(task_name)
        decoder_io["target_type"] = "multi_regulon_compact"
        return decoder_io

    # ------------------------------------------------------------------
    # Legacy one-regulon pair format (compatibility only)
    # ------------------------------------------------------------------

    def parse_regulon_content(self, content: Any) -> Tuple[int, List[int]]:
        if content is None:
            raise ValueError("regulon_target_content 为空。")
        text = str(content).strip()
        if not text or text.count("->") != 1:
            raise ValueError(f"regulon_target_content 格式错误：{text!r}")
        tf_part, target_part = text.split("->", 1)
        if not tf_part.strip() or not target_part.strip():
            raise ValueError(f"regulon_target_content 缺少 TF 或 targets：{text!r}")
        tf_id = self._validate_global_gene_id(int(tf_part.strip()), "TF")
        target_ids = [
            self._validate_global_gene_id(int(x.strip()), "Target")
            for x in target_part.split(",")
            if x.strip()
        ]
        if not target_ids:
            raise ValueError(f"regulon_target_content 没有有效 target：{text!r}")
        return tf_id, target_ids

    def encode_gene_pairs_body(self, content: Any) -> Tuple[List[int], List[str]]:
        tf_id, target_ids = self.parse_regulon_content(content)
        body: List[int] = []
        for i, target_id in enumerate(target_ids):
            body.extend([tf_id, self.arrow_token_id, target_id])
            if i < len(target_ids) - 1:
                body.append(self.regulon_sep_token_id)
        return body, []

    # ------------------------------------------------------------------
    # Generic target / teacher forcing
    # ------------------------------------------------------------------

    def build_generic_target_ids(
        self,
        task_name: str,
        content: Any,
        target_type: str,
    ) -> Dict[str, Any]:
        target_type = str(target_type).lower().strip()
        task_name_ids = self.encode_task_name_to_global_ids(task_name)
        extra: Dict[str, Any] = {}

        if target_type == "plain_text":
            body_ids = self.encode_plain_text_to_global_ids("" if content is None else str(content))
        elif target_type in {"gene_pairs", "regulon_edges"}:
            body_ids, missing = self.encode_gene_pairs_body(content)
            extra["missing_genes"] = missing
        else:
            raise ValueError(
                f"未知 target_type={target_type!r}。正式支持 plain_text；"
                "Regulon 新格式请使用 build_multi_regulon_decoder_io()。"
            )

        target_ids = (
            [self.task_token_id]
            + task_name_ids
            + [self.start_answer_token_id]
            + body_ids
            + [self.eos_token_id]
        )
        return {"target_ids": [int(x) for x in target_ids], **extra}

    def make_decoder_input_and_labels(
        self,
        target_ids: Sequence[int],
        max_length: Optional[int] = None,
        allow_truncation: bool = True,
    ) -> Dict[str, List[int]]:
        target_ids = [int(x) for x in target_ids]
        max_length = self.max_length if max_length is None else int(max_length)
        if max_length <= 0:
            raise ValueError(f"max_length 必须 > 0，当前为 {max_length}。")
        if not target_ids:
            target_ids = [self.eos_token_id]

        if len(target_ids) > max_length:
            if not allow_truncation:
                raise ValueError(
                    f"结构化 Decoder target 长度 {len(target_ids)} 超过 max_length={max_length}。"
                    "请在 data.py 中使用 Dynamic Target Budget 缩减 targets，禁止 tokenizer 静默截断。"
                )
            target_ids = target_ids[:max_length]
            target_ids[-1] = self.eos_token_id

        decoder_input_ids = [self.bos_token_id] + target_ids[:-1]
        decoder_labels = list(target_ids)
        decoder_attention_mask = [1] * len(decoder_input_ids)

        if self.start_answer_token_id in target_ids:
            prefix_len = target_ids.index(self.start_answer_token_id) + 1
            decoder_labels[:prefix_len] = [-100] * prefix_len

        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_labels": decoder_labels,
            "decoder_attention_mask": decoder_attention_mask,
            "full_ids": target_ids,
            "text_input_ids": decoder_input_ids,
            "text_labels": decoder_labels,
            "text_attention_mask": decoder_attention_mask,
        }

    def build_generic_task_decoder_io(
        self,
        task_name: str,
        content: Any,
        target_type: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        target_pack = self.build_generic_target_ids(task_name, content, target_type)
        decoder_io = self.make_decoder_input_and_labels(
            target_pack["target_ids"],
            max_length=max_length,
            allow_truncation=True,
        )
        for key, value in target_pack.items():
            if key != "target_ids":
                decoder_io[key] = value
        decoder_io["task_name"] = str(task_name)
        decoder_io["target_type"] = str(target_type)
        return decoder_io

    def prepare_batch_decoder_io(
        self,
        items: Sequence[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(items)
        max_len = max((len(x["decoder_input_ids"]) for x in items), default=0)
        decoder_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        decoder_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        decoder_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, item in enumerate(items):
            cur_len = len(item["decoder_input_ids"])
            decoder_input_ids[i, :cur_len] = torch.tensor(item["decoder_input_ids"], dtype=torch.long)
            decoder_labels[i, :cur_len] = torch.tensor(item["decoder_labels"], dtype=torch.long)
            decoder_attention_mask[i, :cur_len] = torch.tensor(item["decoder_attention_mask"], dtype=torch.long)
        if device is not None:
            decoder_input_ids = decoder_input_ids.to(device)
            decoder_labels = decoder_labels.to(device)
            decoder_attention_mask = decoder_attention_mask.to(device)
        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_labels": decoder_labels,
            "decoder_attention_mask": decoder_attention_mask,
            "text_input_ids": decoder_input_ids,
            "text_labels": decoder_labels,
            "text_attention_mask": decoder_attention_mask,
        }

    def prepare_batch_mixed_decoder_io(
        self,
        items: Sequence[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.prepare_batch_decoder_io(items=items, device=device)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def ids_to_debug_tokens(self, ids: Sequence[int]) -> List[str]:
        out: List[str] = []
        for gid in ids:
            gid = int(gid)
            token = self.global_id_to_token.get(gid, f"<missing:{gid}>")
            if self.global_id_is_gene(gid):
                symbol = self.global_id_to_gene_symbol.get(gid, token)
                out.append(f"<gene:{symbol}>")
            elif self.global_id_is_text(gid):
                prefix_len = len(self.text_token_prefix)
                suffix_len = len(self.token_suffix)
                out.append(token[prefix_len:-suffix_len] if suffix_len > 0 else token[prefix_len:])
            else:
                out.append(token)
        return out

    def decode_debug(self, ids: Sequence[int]) -> str:
        return " ".join(self.ids_to_debug_tokens(ids))

    def __repr__(self) -> str:
        return (
            "GlobalGeneTextTokenizer("
            f"vocab_size={self.vocab_size}, max_length={self.max_length}, "
            f"n_genes={len(self.global_id_to_gene_symbol)})"
        )


# Lightweight CLI tokenizer check.
def str2bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔值参数：{value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="测试 GlobalGeneTextTokenizer。")
    p.add_argument("--text_tokenizer_path", required=True)
    p.add_argument("--global_vocab_path", required=True)
    p.add_argument("--global_vocab_meta_path", required=True)
    p.add_argument("--gene_table_path", required=True)
    p.add_argument("--max_length", type=int, default=1025)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tok = GlobalGeneTextTokenizer.from_files(
        text_tokenizer_path=args.text_tokenizer_path,
        global_vocab_path=args.global_vocab_path,
        global_vocab_meta_path=args.global_vocab_meta_path,
        gene_table_path=args.gene_table_path,
        max_length=args.max_length,
    )
    print(tok)


if __name__ == "__main__":
    main()
