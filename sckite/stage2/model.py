#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


PathLike = Union[str, Path]


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return preds.sum() * 0.0
    return F.mse_loss(preds[mask], targets[mask].float(), reduction="mean")


def decoder_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=int(ignore_index))


def masked_gene_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask].long(), ignore_index=int(ignore_index))


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    preds = logits.argmax(dim=-1)
    valid_mask = labels.ne(int(ignore_index))
    if valid_mask.sum().item() == 0:
        return float("nan")
    acc = preds.eq(labels).logical_and(valid_mask).sum().float() / valid_mask.sum().float()
    return float(acc.item())


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = values.float().unsqueeze(-1)
        return self.proj(x)


class PreNormSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model) * int(expansion_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        h = self.norm1(x)
        attn_out, attn_weights = self.self_attn(query=h, key=h, value=h, key_padding_mask=key_padding_mask, need_weights=return_attn, average_attn_weights=False)
        x = x + self.dropout1(attn_out)
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        if return_attn:
            return x, attn_weights
        return x


class DenseTXEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([PreNormSelfAttentionBlock(d_model=d_model, n_heads=n_heads, expansion_ratio=expansion_ratio, dropout=dropout) for _ in range(int(n_layers))])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, return_last_attn: bool = False):
        last_attn = None
        for i, block in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if return_last_attn and is_last:
                x, last_attn = block(x, key_padding_mask=key_padding_mask, return_attn=True)
            else:
                x = block(x, key_padding_mask=key_padding_mask, return_attn=False)
        x = self.final_norm(x)
        if return_last_attn:
            return x, last_attn
        return x


class MaskedValueHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CrossAttentionDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout_self = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        hidden_dim = int(d_model) * int(expansion_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model))
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, self_attn_mask: Optional[torch.Tensor] = None, decoder_key_padding_mask: Optional[torch.Tensor] = None, encoder_key_padding_mask: Optional[torch.Tensor] = None, return_cross_attn: bool = False):
        h = self.norm_self(x)
        self_out, _ = self.self_attn(query=h, key=h, value=h, attn_mask=self_attn_mask, key_padding_mask=decoder_key_padding_mask, need_weights=False)
        x = x + self.dropout_self(self_out)
        h = self.norm_cross(x)
        cross_out, cross_weights = self.cross_attn(query=h, key=encoder_memory, value=encoder_memory, key_padding_mask=encoder_key_padding_mask, need_weights=return_cross_attn, average_attn_weights=False)
        x = x + self.dropout_cross(cross_out)
        h = self.norm_ffn(x)
        x = x + self.dropout_ffn(self.ffn(h))
        if return_cross_attn:
            return x, cross_weights
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionDecoderBlock(d_model=d_model, n_heads=n_heads, expansion_ratio=expansion_ratio, dropout=dropout) for _ in range(int(n_layers))])
        self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, decoder_key_padding_mask: Optional[torch.Tensor] = None, encoder_key_padding_mask: Optional[torch.Tensor] = None, return_last_cross_attn: bool = False):
        seq_len = x.shape[1]
        causal_mask = self.build_causal_mask(seq_len=seq_len, device=x.device)
        last_cross_attn = None
        for i, block in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if return_last_cross_attn and is_last:
                x, last_cross_attn = block(x=x, encoder_memory=encoder_memory, self_attn_mask=causal_mask, decoder_key_padding_mask=decoder_key_padding_mask, encoder_key_padding_mask=encoder_key_padding_mask, return_cross_attn=True)
            else:
                x = block(x=x, encoder_memory=encoder_memory, self_attn_mask=causal_mask, decoder_key_padding_mask=decoder_key_padding_mask, encoder_key_padding_mask=encoder_key_padding_mask, return_cross_attn=False)
        x = self.final_norm(x)
        if return_last_cross_attn:
            return x, last_cross_attn
        return x


class ScKITEStage2Model(nn.Module):
    def __init__(
        self,
        global_vocab_size: Optional[int] = None,
        vocab_size: Optional[int] = None,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        decoder_n_layers: Optional[int] = None,
        text_decoder_layers: Optional[int] = None,
        decoder_n_heads: Optional[int] = None,
        text_decoder_heads: Optional[int] = None,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        mask_value: float = -3.0,
        max_decoder_length: int = 128,
        value_hidden_dim: int = 128,
        value_head_hidden_dim: int = 256,
        tie_lm_head: bool = True,
        use_mask_flag_embedding: bool = True,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()
        if global_vocab_size is None:
            global_vocab_size = vocab_size
        if global_vocab_size is None:
            raise ValueError("必须传入 global_vocab_size 或 vocab_size。")
        self.global_vocab_size = int(global_vocab_size)
        self.d_model = int(d_model)
        self.pad_token_id = int(pad_token_id)
        self.mask_value = float(mask_value)
        self.max_decoder_length = int(max_decoder_length)
        self.use_mask_flag_embedding = bool(use_mask_flag_embedding)
        decoder_layers = int(decoder_n_layers if decoder_n_layers is not None else (text_decoder_layers if text_decoder_layers is not None else 2))
        decoder_heads = int(decoder_n_heads if decoder_n_heads is not None else (text_decoder_heads if text_decoder_heads is not None else n_heads))
        self.shared_token_embedding = nn.Embedding(self.global_vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.value_encoder = ContinuousValueEncoder(d_model=self.d_model, hidden_dim=value_hidden_dim, dropout=dropout)
        self.mask_flag_embedding = nn.Embedding(2, self.d_model) if self.use_mask_flag_embedding else None
        self.encoder = DenseTXEncoder(d_model=self.d_model, n_heads=n_heads, n_layers=n_layers, expansion_ratio=expansion_ratio, dropout=dropout)
        self.value_head = MaskedValueHead(d_model=self.d_model, hidden_dim=value_head_hidden_dim, dropout=dropout)


        self.regulon_decoder_position_embedding = nn.Embedding(self.max_decoder_length, self.d_model)
        self.regulon_decoder_token_norm = nn.LayerNorm(self.d_model)
        self.regulon_decoder_dropout = nn.Dropout(dropout)
        self.regulon_decoder = CrossAttentionDecoder(
            d_model=self.d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
        )


        self.annotation_decoder_position_embedding = nn.Embedding(self.max_decoder_length, self.d_model)
        self.annotation_decoder_token_norm = nn.LayerNorm(self.d_model)
        self.annotation_decoder_dropout = nn.Dropout(dropout)
        self.annotation_decoder = CrossAttentionDecoder(
            d_model=self.d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
        )


        self.unified_lm_head = nn.Linear(self.d_model, self.global_vocab_size, bias=False)
        if bool(tie_lm_head):
            self.unified_lm_head.weight = self.shared_token_embedding.weight



        self.regulon_lm_head = nn.Linear(self.d_model, self.global_vocab_size, bias=False)
        self.annotation_lm_head = nn.Linear(self.d_model, self.global_vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.shared_token_embedding.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None and 0 <= self.pad_token_id < self.shared_token_embedding.weight.shape[0]:
            with torch.no_grad():
                self.shared_token_embedding.weight[self.pad_token_id].zero_()
        nn.init.normal_(self.regulon_decoder_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.annotation_decoder_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.regulon_lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.annotation_lm_head.weight, mean=0.0, std=0.02)
        if self.mask_flag_embedding is not None:
            nn.init.normal_(self.mask_flag_embedding.weight, mean=0.0, std=0.02)

    def encode(self, encoder_input_gene_ids: torch.Tensor, encoder_input_values: torch.Tensor, encoder_key_padding_mask: Optional[torch.Tensor] = None, return_last_attn: bool = False):
        token_emb = self.shared_token_embedding(encoder_input_gene_ids.long())
        token_emb = self.token_norm(token_emb)
        value_emb = self.value_encoder(encoder_input_values.float())
        x = token_emb + value_emb
        if self.mask_flag_embedding is not None:
            mask_flags = encoder_input_values.eq(self.mask_value).long().clamp(min=0, max=1)
            x = x + self.mask_flag_embedding(mask_flags)
        if return_last_attn:
            encoder_outputs, last_attn = self.encoder(x, key_padding_mask=encoder_key_padding_mask, return_last_attn=True)
        else:
            encoder_outputs = self.encoder(x, key_padding_mask=encoder_key_padding_mask, return_last_attn=False)
            last_attn = None
        return encoder_outputs, last_attn

    def pool_cell_embedding(self, encoder_outputs: torch.Tensor, encoder_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if encoder_outputs.shape[1] > 0:
            return encoder_outputs[:, 0, :]
        if encoder_key_padding_mask is None:
            return encoder_outputs.mean(dim=1)
        valid = (~encoder_key_padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (encoder_outputs * valid).sum(dim=1) / denom

    def _decode_task(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        position_embedding: nn.Embedding,
        token_norm: nn.LayerNorm,
        decoder_dropout: nn.Dropout,
        decoder: CrossAttentionDecoder,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
        return_last_cross_attn: bool = False,
    ):
        """Shared execution logic for one task-specific decoder branch."""
        batch_size, seq_len = decoder_input_ids.shape
        if seq_len > self.max_decoder_length:
            raise ValueError(
                f"decoder 序列长度 {seq_len} 超过 max_decoder_length={self.max_decoder_length}，请增大模型配置。"
            )

        pos_ids = torch.arange(
            seq_len, device=decoder_input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)



        x = self.shared_token_embedding(decoder_input_ids.long()) + position_embedding(pos_ids)
        x = token_norm(x)
        x = decoder_dropout(x)

        decoder_key_padding_mask = None
        if decoder_attention_mask is not None:
            decoder_key_padding_mask = decoder_attention_mask.eq(0)

        if return_last_cross_attn:
            decoder_hidden, last_cross_attn = decoder(
                x=x,
                encoder_memory=encoder_outputs,
                decoder_key_padding_mask=decoder_key_padding_mask,
                encoder_key_padding_mask=encoder_key_padding_mask,
                return_last_cross_attn=True,
            )
        else:
            decoder_hidden = decoder(
                x=x,
                encoder_memory=encoder_outputs,
                decoder_key_padding_mask=decoder_key_padding_mask,
                encoder_key_padding_mask=encoder_key_padding_mask,
                return_last_cross_attn=False,
            )
            last_cross_attn = None

        return decoder_hidden, last_cross_attn

    def decode_regulon(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
        return_last_cross_attn: bool = False,
    ):
        """Run the regulon-specific autoregressive cross-attention decoder."""
        return self._decode_task(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            position_embedding=self.regulon_decoder_position_embedding,
            token_norm=self.regulon_decoder_token_norm,
            decoder_dropout=self.regulon_decoder_dropout,
            decoder=self.regulon_decoder,
            decoder_attention_mask=decoder_attention_mask,
            encoder_key_padding_mask=encoder_key_padding_mask,
            return_last_cross_attn=return_last_cross_attn,
        )

    def decode_annotation(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
        return_last_cross_attn: bool = False,
    ):
        """Run the annotation-specific autoregressive cross-attention decoder."""
        return self._decode_task(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            position_embedding=self.annotation_decoder_position_embedding,
            token_norm=self.annotation_decoder_token_norm,
            decoder_dropout=self.annotation_decoder_dropout,
            decoder=self.annotation_decoder,
            decoder_attention_mask=decoder_attention_mask,
            encoder_key_padding_mask=encoder_key_padding_mask,
            return_last_cross_attn=return_last_cross_attn,
        )

    def forward(
        self,
        encoder_input_gene_ids: Optional[torch.Tensor] = None,
        encoder_input_values: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_task_ids: Optional[torch.Tensor] = None,
        genes: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        return_encoder_gene_logits: bool = False,
        return_last_attn: bool = False,
        return_last_cross_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if encoder_input_gene_ids is None:
            encoder_input_gene_ids = genes
        if encoder_input_values is None:
            encoder_input_values = input_values
        if decoder_input_ids is None:
            decoder_input_ids = text_input_ids
        if decoder_attention_mask is None:
            decoder_attention_mask = text_attention_mask
        if encoder_input_gene_ids is None or encoder_input_values is None:
            raise ValueError("必须提供 encoder_input_gene_ids/encoder_input_values 或 genes/input_values。")
        if encoder_key_padding_mask is None:
            encoder_key_padding_mask = encoder_input_gene_ids.eq(self.pad_token_id)
        encoder_outputs, encoder_last_attn = self.encode(encoder_input_gene_ids=encoder_input_gene_ids, encoder_input_values=encoder_input_values, encoder_key_padding_mask=encoder_key_padding_mask, return_last_attn=return_last_attn)
        expr_preds = self.value_head(encoder_outputs)
        cell_emb = self.pool_cell_embedding(encoder_outputs, encoder_key_padding_mask=encoder_key_padding_mask)
        gene_logits = self.unified_lm_head(encoder_outputs) if return_encoder_gene_logits else None

        regulon_hidden = None
        regulon_logits = None
        regulon_batch_indices = None
        regulon_last_cross_attn = None

        annotation_hidden = None
        annotation_logits = None
        annotation_batch_indices = None
        annotation_last_cross_attn = None

        if decoder_input_ids is not None:
            if decoder_task_ids is None:
                raise ValueError("提供 decoder_input_ids 时必须同时提供 decoder_task_ids（0=regulon，1=annotation）。")
            if decoder_task_ids.ndim != 1 or decoder_task_ids.shape[0] != decoder_input_ids.shape[0]:
                raise ValueError(
                    "decoder_task_ids 必须是形状 [batch_size] 的一维张量，并与 decoder_input_ids 的 batch 维一致。"
                )

            decoder_task_ids = decoder_task_ids.to(decoder_input_ids.device).long()
            valid_task_mask = decoder_task_ids.eq(0) | decoder_task_ids.eq(1)
            if not valid_task_mask.all():
                bad_ids = torch.unique(decoder_task_ids[~valid_task_mask]).detach().cpu().tolist()
                raise ValueError(f"发现未知 decoder_task_ids={bad_ids}；当前仅支持 0=regulon、1=annotation。")


            regulon_mask = decoder_task_ids.eq(0)
            if regulon_mask.any():
                regulon_batch_indices = torch.where(regulon_mask)[0]
                regulon_hidden, regulon_last_cross_attn = self.decode_regulon(
                    decoder_input_ids=decoder_input_ids[regulon_mask],
                    encoder_outputs=encoder_outputs[regulon_mask],
                    decoder_attention_mask=(
                        decoder_attention_mask[regulon_mask]
                        if decoder_attention_mask is not None
                        else None
                    ),
                    encoder_key_padding_mask=encoder_key_padding_mask[regulon_mask],
                    return_last_cross_attn=return_last_cross_attn,
                )
                regulon_logits = self.regulon_lm_head(regulon_hidden)


            annotation_mask = decoder_task_ids.eq(1)
            if annotation_mask.any():
                annotation_batch_indices = torch.where(annotation_mask)[0]
                annotation_hidden, annotation_last_cross_attn = self.decode_annotation(
                    decoder_input_ids=decoder_input_ids[annotation_mask],
                    encoder_outputs=encoder_outputs[annotation_mask],
                    decoder_attention_mask=(
                        decoder_attention_mask[annotation_mask]
                        if decoder_attention_mask is not None
                        else None
                    ),
                    encoder_key_padding_mask=encoder_key_padding_mask[annotation_mask],
                    return_last_cross_attn=return_last_cross_attn,
                )
                annotation_logits = self.annotation_lm_head(annotation_hidden)

        return {
            "encoder_outputs": encoder_outputs,
            "cell_emb": cell_emb,
            "expr_preds": expr_preds,
            "gene_logits": gene_logits,
            "encoder_last_attn": encoder_last_attn,
            "mlm_output": expr_preds,
            "regulon_hidden": regulon_hidden,
            "regulon_logits": regulon_logits,
            "regulon_batch_indices": regulon_batch_indices,
            "regulon_last_cross_attn": regulon_last_cross_attn,
            "annotation_hidden": annotation_hidden,
            "annotation_logits": annotation_logits,
            "annotation_batch_indices": annotation_batch_indices,
            "annotation_last_cross_attn": annotation_last_cross_attn,
        }




def freeze_encoder_backbone(model: ScKITEStage2Model, freeze_embeddings: bool = False, freeze_value_head: bool = False) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.value_encoder.parameters():
        param.requires_grad = False
    if model.mask_flag_embedding is not None:
        for param in model.mask_flag_embedding.parameters():
            param.requires_grad = False
    if freeze_embeddings:
        for param in model.shared_token_embedding.parameters():
            param.requires_grad = False
        for param in model.token_norm.parameters():
            param.requires_grad = False
    if freeze_value_head:
        for param in model.value_head.parameters():
            param.requires_grad = False






from pathlib import Path as _PathForStage1Loader
import os as _os_for_stage1_loader
import json as _json_for_stage1_loader
import yaml as _yaml_for_stage1_loader
import torch as _torch_for_stage1_loader


def _read_json_or_jsonl_for_stage1_loader(path):
    path = _PathForStage1Loader(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [_json_for_stage1_loader.loads(line) for line in text.splitlines() if line.strip()]
    try:
        return _json_for_stage1_loader.loads(text)
    except Exception:
        return [_json_for_stage1_loader.loads(line) for line in text.splitlines() if line.strip()]


def _build_token_to_id_for_stage1_loader(vocab_obj):
    if isinstance(vocab_obj, dict) and "token_to_id" in vocab_obj:
        return {str(k): int(v) for k, v in vocab_obj["token_to_id"].items()}
    if isinstance(vocab_obj, dict) and "id_to_token" in vocab_obj:
        return {str(v): int(k) for k, v in vocab_obj["id_to_token"].items()}
    if isinstance(vocab_obj, dict):
        if all(isinstance(v, int) for v in vocab_obj.values()):
            return {str(k): int(v) for k, v in vocab_obj.items()}
        if all(str(k).isdigit() for k in vocab_obj.keys()):
            return {str(v): int(k) for k, v in vocab_obj.items()}
    if isinstance(vocab_obj, list):
        out = {}
        for item in vocab_obj:
            if not isinstance(item, dict):
                continue
            tid = item.get("token_id", item.get("id", item.get("global_id", None)))
            tok = item.get("global_token", item.get("token", item.get("gene_symbol", item.get("ensembl_id", None))))
            if tid is not None and tok is not None:
                out[str(tok)] = int(tid)
        return out
    return {}


def _build_global_token_to_id_for_stage1_loader(global_vocab_path):
    obj = _read_json_or_jsonl_for_stage1_loader(global_vocab_path)
    return _build_token_to_id_for_stage1_loader(obj)


def _find_stage2_yaml_for_stage1_loader():
    module_dir = _PathForStage1Loader(__file__).resolve().parent
    candidates = [
        _os_for_stage1_loader.environ.get("STAGE2_CONFIG_PATH", ""),
        module_dir / "config.yaml",
        module_dir.parent / "stage2" / "config.yaml",
    ]
    for p in candidates:
        if p and _PathForStage1Loader(p).exists():
            return _PathForStage1Loader(p)
    return None


def _get_vocab_paths_for_stage1_loader():
    yaml_path = _find_stage2_yaml_for_stage1_loader()
    cfg = {}
    if yaml_path is not None:
        cfg = _yaml_for_stage1_loader.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    gv_cfg = cfg.get("global_vocab", {}) if isinstance(cfg, dict) else {}

    global_vocab_path = _os_for_stage1_loader.environ.get("GLOBAL_VOCAB_PATH") or gv_cfg.get("global_vocab_path")
    gene_table_path = _os_for_stage1_loader.environ.get("GENE_TABLE_PATH") or gv_cfg.get("gene_table_path")
    stage1_vocab_path = (
        _os_for_stage1_loader.environ.get("STAGE1_VOCAB_PATH")
        or gv_cfg.get("stage1_vocab_path")
        or gv_cfg.get("raw_gene_vocab_path")
        or gv_cfg.get("old_gene_vocab_path")
    )
    return global_vocab_path, gene_table_path, stage1_vocab_path


def _select_stage1_embedding_key_for_stage1_loader(state, own_state):
    candidate_keys = [
        "gene_encoder.embedding.weight",
        "gene_embedding.weight",
        "token_embedding.weight",
        "gene_token_embedding.weight",
        "embedding.weight",
        "shared_token_embedding.weight",
    ]
    for k in candidate_keys:
        if k in state and "shared_token_embedding.weight" in own_state:
            if state[k].ndim == 2 and state[k].shape[1] == own_state["shared_token_embedding.weight"].shape[1]:
                return k
    for k, v in state.items():
        lk = k.lower()
        if "embedding" in lk and hasattr(v, "shape") and v.ndim == 2:
            if "shared_token_embedding.weight" in own_state and v.shape[1] == own_state["shared_token_embedding.weight"].shape[1]:
                return k
    return None


def _copy_embedding_by_gene_table_for_stage1_loader(model, state, copied_info, verbose=True):
    own_state = model.state_dict()
    if "shared_token_embedding.weight" not in own_state:
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": "no shared_token_embedding.weight"}
        return

    global_vocab_path, gene_table_path, stage1_vocab_path = _get_vocab_paths_for_stage1_loader()
    if global_vocab_path is None or gene_table_path is None:
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": "missing global_vocab_path or gene_table_path"}
        return

    global_vocab_path = _PathForStage1Loader(global_vocab_path)
    gene_table_path = _PathForStage1Loader(gene_table_path)
    if not global_vocab_path.exists() or not gene_table_path.exists():
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": f"path not exists: {global_vocab_path}, {gene_table_path}"}
        return

    global_token_to_id = _build_global_token_to_id_for_stage1_loader(global_vocab_path)
    gene_table = _read_json_or_jsonl_for_stage1_loader(gene_table_path)
    if isinstance(gene_table, dict):
        gene_table = gene_table.get("genes", gene_table.get("items", gene_table.get("data", [])))

    old_token_to_id = {}
    if stage1_vocab_path is not None and _PathForStage1Loader(stage1_vocab_path).exists():
        old_token_to_id = _build_token_to_id_for_stage1_loader(_read_json_or_jsonl_for_stage1_loader(stage1_vocab_path))

    old_embed_key = _select_stage1_embedding_key_for_stage1_loader(state, own_state)
    if old_embed_key is None:
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": "no stage1 embedding key found"}
        return

    old_embed = state[old_embed_key]
    new_embed = own_state["shared_token_embedding.weight"]
    copied = 0
    skipped = 0
    examples = []

    with _torch_for_stage1_loader.no_grad():
        for item in gene_table:
            if not isinstance(item, dict):
                skipped += 1
                continue

            old_id = item.get("token_id", item.get("stage1_token_id", item.get("old_token_id", None)))
            gene_symbol = item.get("gene_symbol", item.get("symbol", None))
            ensembl_id = item.get("ensembl_id", item.get("ensembl", None))
            global_id = item.get("global_id", item.get("global_token_id", item.get("new_token_id", None)))
            global_token = item.get("global_token", None)

            if old_id is None and old_token_to_id:
                for tok in [gene_symbol, ensembl_id, f"<gene:{gene_symbol}>" if gene_symbol else None, f"<gene:{ensembl_id}>" if ensembl_id else None]:
                    if tok is not None and str(tok) in old_token_to_id:
                        old_id = old_token_to_id[str(tok)]
                        break

            if global_id is None:
                candidates = []
                if global_token is not None:
                    candidates.append(str(global_token))
                if gene_symbol is not None:
                    candidates.append(f"<gene:{gene_symbol}>")
                if ensembl_id is not None:
                    candidates.append(f"<gene:{ensembl_id}>")
                for tok in candidates:
                    if tok in global_token_to_id:
                        global_id = global_token_to_id[tok]
                        global_token = tok
                        break

            if old_id is None or global_id is None:
                skipped += 1
                continue
            old_id = int(old_id)
            global_id = int(global_id)
            if old_id < 0 or old_id >= old_embed.shape[0] or global_id < 0 or global_id >= new_embed.shape[0]:
                skipped += 1
                continue

            new_embed[global_id].copy_(old_embed[old_id])
            copied += 1
            if len(examples) < 10:
                examples.append({"old_id": old_id, "global_id": global_id, "global_token": global_token, "gene_symbol": gene_symbol, "ensembl_id": ensembl_id})

    copied_info["embedding_by_gene_table"] = {
        "copied": copied,
        "skipped": skipped,
        "old_embed_key": old_embed_key,
        "global_vocab_path": str(global_vocab_path),
        "gene_table_path": str(gene_table_path),
        "stage1_vocab_path": str(stage1_vocab_path) if stage1_vocab_path else None,
        "examples": examples,
    }
    if verbose:
        print(f"[vocab-aware embedding copy] copied={copied}, skipped={skipped}, old_embed_key={old_embed_key}")
        print("[vocab-aware examples]", examples[:5])


def _copy_special_embeddings_by_exact_token_for_stage1_loader(model, state, copied_info, verbose=True):
    own_state = model.state_dict()
    global_vocab_path, gene_table_path, stage1_vocab_path = _get_vocab_paths_for_stage1_loader()
    if stage1_vocab_path is None or global_vocab_path is None:
        copied_info["special_embedding_by_name"] = {"copied": 0, "reason": "missing stage1_vocab_path or global_vocab_path"}
        return
    if not _PathForStage1Loader(stage1_vocab_path).exists() or not _PathForStage1Loader(global_vocab_path).exists():
        copied_info["special_embedding_by_name"] = {"copied": 0, "reason": "path not exists"}
        return

    old_token_to_id = _build_token_to_id_for_stage1_loader(_read_json_or_jsonl_for_stage1_loader(stage1_vocab_path))
    global_token_to_id = _build_global_token_to_id_for_stage1_loader(global_vocab_path)
    old_embed_key = _select_stage1_embedding_key_for_stage1_loader(state, own_state)
    if old_embed_key is None or "shared_token_embedding.weight" not in own_state:
        copied_info["special_embedding_by_name"] = {"copied": 0, "reason": "missing embedding"}
        return

    old_embed = state[old_embed_key]
    new_embed = own_state["shared_token_embedding.weight"]
    allowed_specials = ["<cls>", "<pad>"]
    copied = 0
    examples = []
    with _torch_for_stage1_loader.no_grad():
        for tok in allowed_specials:
            if tok in old_token_to_id and tok in global_token_to_id:
                old_id = int(old_token_to_id[tok])
                new_id = int(global_token_to_id[tok])
                if old_id < old_embed.shape[0] and new_id < new_embed.shape[0]:
                    new_embed[new_id].copy_(old_embed[old_id])
                    copied += 1
                    examples.append({"token": tok, "old_id": old_id, "new_id": new_id})
    copied_info["special_embedding_by_name"] = {"copied": copied, "examples": examples}
    if verbose and copied > 0:
        print(f"[special embedding copy] copied={copied}, examples={examples}")


def load_stage1_encoder_weights(model, ckpt_path, strict_encoder: bool = False, verbose: bool = True):
    """
    从 Stage1 checkpoint 加载 Encoder 相关权重。

    优先级：
    1. 如果 Stage1 embedding 与当前 shared_token_embedding shape 完全一致，
       直接整块加载 shared_token_embedding.weight。
    2. 如果 shape 不一致，回退到 gene_table/global_vocab 映射。
    3. 其余 Encoder 相关同名同 shape 参数直接加载。
    4. 兼容旧 attention 参数名 attn -> self_attn。

    Decoder 参数不会从 Stage1 checkpoint 加载。
    """
    ckpt_path = _PathForStage1Loader(ckpt_path)
    ckpt = _torch_for_stage1_loader.load(str(ckpt_path), map_location="cpu")
    state = (
        ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if isinstance(ckpt, dict)
        else ckpt
    )
    own_state = model.state_dict()

    load_state = {}
    copied_info = {
        "direct": [],
        "mapped": [],
        "skipped": [],
        "embedding_direct": {
            "loaded": False,
            "from": None,
            "to": "shared_token_embedding.weight",
            "shape": None,
        },
    }


    stage1_embed_key = _select_stage1_embedding_key_for_stage1_loader(
        state,
        own_state,
    )

    if (
        stage1_embed_key is not None
        and "shared_token_embedding.weight" in own_state
    ):
        old_embedding = state[stage1_embed_key]
        new_embedding = own_state["shared_token_embedding.weight"]

        if tuple(old_embedding.shape) == tuple(new_embedding.shape):
            load_state["shared_token_embedding.weight"] = old_embedding
            copied_info["embedding_direct"] = {
                "loaded": True,
                "from": stage1_embed_key,
                "to": "shared_token_embedding.weight",
                "shape": tuple(old_embedding.shape),
            }

            if verbose:
                print(
                    "[embedding direct load] "
                    f"from={stage1_embed_key} "
                    "-> shared_token_embedding.weight, "
                    f"shape={tuple(old_embedding.shape)}"
                )


    for key, value in state.items():
        new_key = key.replace("module.", "")

        if new_key == stage1_embed_key:
            continue
        if new_key == "shared_token_embedding.weight":
            continue

        if (
            new_key in own_state
            and own_state[new_key].shape == value.shape
        ):
            load_state[new_key] = value
            copied_info["direct"].append(new_key)
            continue

        if (
            new_key.startswith("encoder.layers.")
            and ".attn." in new_key
        ):
            mapped_key = new_key.replace(
                ".attn.",
                ".self_attn.",
            )
            if (
                mapped_key in own_state
                and own_state[mapped_key].shape == value.shape
            ):
                load_state[mapped_key] = value
                copied_info["mapped"].append(
                    {
                        "from": new_key,
                        "to": mapped_key,
                        "shape": tuple(value.shape),
                    }
                )
                continue

        copied_info["skipped"].append(new_key)

    missing, unexpected = model.load_state_dict(
        load_state,
        strict=False,
    )

    copied_info["missing_after_partial_load"] = list(missing)
    copied_info["unexpected_after_partial_load"] = list(unexpected)


    if not copied_info["embedding_direct"]["loaded"]:
        _copy_embedding_by_gene_table_for_stage1_loader(
            model,
            state,
            copied_info,
            verbose=verbose,
        )
        _copy_special_embeddings_by_exact_token_for_stage1_loader(
            model,
            state,
            copied_info,
            verbose=verbose,
        )
    else:
        copied_info["embedding_by_gene_table"] = {
            "copied": 0,
            "reason": "embedding loaded directly because shapes are identical",
        }
        copied_info["special_embedding_by_name"] = {
            "copied": 0,
            "reason": "special tokens included in direct embedding load",
        }

    if (
        strict_encoder
        and len(copied_info["direct"]) == 0
        and len(copied_info["mapped"]) == 0
        and not copied_info["embedding_direct"]["loaded"]
    ):
        raise RuntimeError(
            f"没有从 {ckpt_path} 加载到任何可用 Stage1 权重。"
        )

    if verbose:
        n_attn = sum(
            1
            for x in copied_info["mapped"]
            if ".self_attn." in str(x)
        )
        emb_info = copied_info.get(
            "embedding_by_gene_table",
            {},
        )
        sp_info = copied_info.get(
            "special_embedding_by_name",
            {},
        )
        direct_embed = copied_info["embedding_direct"]

        print(
            "[vocab-aware load_stage1_encoder_weights] "
            f"direct={len(copied_info['direct'])}, "
            f"mapped={len(copied_info['mapped'])}, "
            f"attn_mapped={n_attn}, "
            f"embedding_direct={direct_embed['loaded']}, "
            f"embedding_from={direct_embed['from']}, "
            f"embedding_shape={direct_embed['shape']}, "
            f"gene_embedding_copied={emb_info.get('copied', 0)}, "
            f"special_copied={sp_info.get('copied', 0)}, "
            f"skipped={len(copied_info['skipped'])}"
        )
        print(
            "[mapped examples]",
            copied_info["mapped"][:8],
        )

    return copied_info
