#!/usr/bin/env python3  # 指定使用 Python3 解释器运行。 #
# -*- coding: utf-8 -*-  # 指定源码编码为 UTF-8。 #

from __future__ import annotations  # 启用延迟类型注解，避免前向引用报错。 #

from pathlib import Path  # 导入 Path，用于处理 checkpoint 路径。 #
from typing import Dict, Optional, Tuple, Union, Any  # 导入类型注解。 #

import torch  # 导入 PyTorch 主库。 #
import torch.nn as nn  # 导入神经网络模块。 #
import torch.nn.functional as F  # 导入常用函数接口。 #


PathLike = Union[str, Path]  # 定义路径类型别名。 #


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # 定义 masked expression MSE 损失。 #
    mask = mask.bool()  # 确保 mask 是 bool 类型。 #
    if mask.sum() == 0:  # 如果当前 batch 没有任何有效 mask 位置。 #
        return preds.sum() * 0.0
    return F.mse_loss(preds[mask], targets[mask].float(), reduction="mean")


def decoder_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:  # 定义 decoder next-token CE 损失。 #
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=int(ignore_index))  # 展平后计算交叉熵。 #


def masked_gene_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask].long(), ignore_index=int(ignore_index))


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:  # 计算 token 级准确率。 #
    preds = logits.argmax(dim=-1)  # 取最大 logit 对应的 token id。 #
    valid_mask = labels.ne(int(ignore_index))  # 构建有效 label mask。 #
    if valid_mask.sum().item() == 0:  # 如果没有有效 label。 #
        return float("nan")  # 返回 NaN。 #
    acc = preds.eq(labels).logical_and(valid_mask).sum().float() / valid_mask.sum().float()  # 计算准确率。 #
    return float(acc.item())  # 转成 Python float。 #


class ContinuousValueEncoder(nn.Module):  # 定义连续表达值编码器。 #
    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:  # 初始化表达值编码器。 #
        super().__init__()  # 调用父类初始化。 #
        self.proj = nn.Sequential(  # 定义将标量表达值映射到 d_model 的小 MLP。 #
            nn.Linear(1, hidden_dim),  # 第一层线性映射。 #
            nn.GELU(),  # GELU 激活函数。 #
            nn.Dropout(dropout),  # Dropout。 #
            nn.Linear(hidden_dim, d_model),  # 第二层映射到 d_model。 #
            nn.LayerNorm(d_model),  # 输出做 LayerNorm。 #
        )  # MLP 定义结束。 #

    def forward(self, values: torch.Tensor) -> torch.Tensor:  # 前向传播。 #
        x = values.float().unsqueeze(-1)  # 将 [B, L] 扩展为 [B, L, 1]。 #
        return self.proj(x)  # 返回 [B, L, D] 表达值嵌入。 #


class PreNormSelfAttentionBlock(nn.Module):  # 定义 encoder 使用的 PreNorm self-attention block。 #
    def __init__(self, d_model: int, n_heads: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:  # 初始化 block。 #
        super().__init__()  # 调用父类初始化。 #
        self.norm1 = nn.LayerNorm(d_model)  # self-attention 前的 LayerNorm。 #
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)  # 多头自注意力。 #
        self.dropout1 = nn.Dropout(dropout)  # attention 输出 dropout。 #
        self.norm2 = nn.LayerNorm(d_model)  # FFN 前的 LayerNorm。 #
        hidden_dim = int(d_model) * int(expansion_ratio)  # 计算 FFN 隐藏层维度。 #
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model))  # 定义 FFN。 #
        self.dropout2 = nn.Dropout(dropout)  # FFN 输出 dropout。 #

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, return_attn: bool = False):  # 前向传播。 #
        h = self.norm1(x)  # 对输入做 PreNorm。 #
        attn_out, attn_weights = self.self_attn(query=h, key=h, value=h, key_padding_mask=key_padding_mask, need_weights=return_attn, average_attn_weights=False)  # 计算 self-attention。 #
        x = x + self.dropout1(attn_out)  # 残差连接。 #
        h = self.norm2(x)  # 对残差结果做 PreNorm。 #
        x = x + self.dropout2(self.ffn(h))  # FFN 残差连接。 #
        if return_attn:  # 如果需要返回 attention。 #
            return x, attn_weights  # 返回输出和 attention 权重。 #
        return x  # 返回输出。 #


class DenseTXEncoder(nn.Module):  # 定义 dense Transformer encoder。 #
    def __init__(self, d_model: int, n_heads: int, n_layers: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:  # 初始化 encoder。 #
        super().__init__()  # 调用父类初始化。 #
        self.layers = nn.ModuleList([PreNormSelfAttentionBlock(d_model=d_model, n_heads=n_heads, expansion_ratio=expansion_ratio, dropout=dropout) for _ in range(int(n_layers))])  # 堆叠 encoder blocks。 #
        self.final_norm = nn.LayerNorm(d_model)  # 最后一层 LayerNorm。 #

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, return_last_attn: bool = False):  # 前向传播。 #
        last_attn = None  # 初始化最后一层 attention。 #
        for i, block in enumerate(self.layers):  # 遍历 encoder block。 #
            is_last = i == len(self.layers) - 1  # 判断是否最后一层。 #
            if return_last_attn and is_last:  # 如果需要最后一层 attention。 #
                x, last_attn = block(x, key_padding_mask=key_padding_mask, return_attn=True)  # 返回 attention。 #
            else:  # 普通前向。 #
                x = block(x, key_padding_mask=key_padding_mask, return_attn=False)  # 不返回 attention。 #
        x = self.final_norm(x)  # 做最终 LayerNorm。 #
        if return_last_attn:  # 如果请求 attention。 #
            return x, last_attn  # 返回 encoder 输出和最后一层 attention。 #
        return x  # 返回 encoder 输出。 #


class MaskedValueHead(nn.Module):  # 定义 expression reconstruction head。 #
    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:  # 初始化 head。 #
        super().__init__()  # 调用父类初始化。 #
        self.net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))  # 定义两层 MLP。 #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播。 #
        return self.net(x).squeeze(-1)  # 返回 [B, L] 表达值预测。 #


class CrossAttentionDecoderBlock(nn.Module):  # 定义带 cross-attention 的 decoder block。 #
    def __init__(self, d_model: int, n_heads: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:  # 初始化 decoder block。 #
        super().__init__()  # 调用父类初始化。 #
        self.norm_self = nn.LayerNorm(d_model)  # causal self-attention 前 LayerNorm。 #
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)  # decoder causal self-attention。 #
        self.dropout_self = nn.Dropout(dropout)  # self-attention dropout。 #
        self.norm_cross = nn.LayerNorm(d_model)  # cross-attention 前 query LayerNorm。 #
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)  # encoder-decoder cross-attention。 #
        self.dropout_cross = nn.Dropout(dropout)  # cross-attention dropout。 #
        self.norm_ffn = nn.LayerNorm(d_model)  # FFN 前 LayerNorm。 #
        hidden_dim = int(d_model) * int(expansion_ratio)  # 计算 FFN 隐藏层维度。 #
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model))  # 定义 FFN。 #
        self.dropout_ffn = nn.Dropout(dropout)  # FFN dropout。 #

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, self_attn_mask: Optional[torch.Tensor] = None, decoder_key_padding_mask: Optional[torch.Tensor] = None, encoder_key_padding_mask: Optional[torch.Tensor] = None, return_cross_attn: bool = False):  # 前向传播。 #
        h = self.norm_self(x)  # 对 decoder hidden 做 PreNorm。 #
        self_out, _ = self.self_attn(query=h, key=h, value=h, attn_mask=self_attn_mask, key_padding_mask=decoder_key_padding_mask, need_weights=False)  # 计算 causal self-attention。 #
        x = x + self.dropout_self(self_out)  # self-attention 残差连接。 #
        h = self.norm_cross(x)  # 对 self-attention 后 hidden 做 PreNorm。 #
        cross_out, cross_weights = self.cross_attn(query=h, key=encoder_memory, value=encoder_memory, key_padding_mask=encoder_key_padding_mask, need_weights=return_cross_attn, average_attn_weights=False)  # 计算 cross-attention。 #
        x = x + self.dropout_cross(cross_out)  # cross-attention 残差连接。 #
        h = self.norm_ffn(x)  # 对 cross-attention 后 hidden 做 PreNorm。 #
        x = x + self.dropout_ffn(self.ffn(h))  # FFN 残差连接。 #
        if return_cross_attn:  # 如果需要返回 cross-attention。 #
            return x, cross_weights  # 返回输出和 cross-attention 权重。 #
        return x  # 返回输出。 #


class CrossAttentionDecoder(nn.Module):  # 定义完整 cross-attention decoder。 #
    def __init__(self, d_model: int, n_heads: int, n_layers: int, expansion_ratio: int = 4, dropout: float = 0.1) -> None:  # 初始化 decoder。 #
        super().__init__()  # 调用父类初始化。 #
        self.layers = nn.ModuleList([CrossAttentionDecoderBlock(d_model=d_model, n_heads=n_heads, expansion_ratio=expansion_ratio, dropout=dropout) for _ in range(int(n_layers))])  # 堆叠 decoder blocks。 #
        self.final_norm = nn.LayerNorm(d_model)  # 最终 LayerNorm。 #

    @staticmethod  # 静态方法装饰器。 #
    def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:  # 构建 causal mask。 #
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)  # 上三角 True 表示未来 token 不可见。 #

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, decoder_key_padding_mask: Optional[torch.Tensor] = None, encoder_key_padding_mask: Optional[torch.Tensor] = None, return_last_cross_attn: bool = False):  # 前向传播。 #
        seq_len = x.shape[1]  # 获取 decoder 序列长度。 #
        causal_mask = self.build_causal_mask(seq_len=seq_len, device=x.device)  # 构建 causal mask。 #
        last_cross_attn = None  # 初始化最后一层 cross-attention。 #
        for i, block in enumerate(self.layers):  # 遍历 decoder blocks。 #
            is_last = i == len(self.layers) - 1  # 判断是否最后一层。 #
            if return_last_cross_attn and is_last:  # 如果需要最后一层 cross-attention。 #
                x, last_cross_attn = block(x=x, encoder_memory=encoder_memory, self_attn_mask=causal_mask, decoder_key_padding_mask=decoder_key_padding_mask, encoder_key_padding_mask=encoder_key_padding_mask, return_cross_attn=True)  # 返回 cross-attention。 #
            else:  # 普通前向。 #
                x = block(x=x, encoder_memory=encoder_memory, self_attn_mask=causal_mask, decoder_key_padding_mask=decoder_key_padding_mask, encoder_key_padding_mask=encoder_key_padding_mask, return_cross_attn=False)  # 不返回 attention。 #
        x = self.final_norm(x)  # 最终归一化。 #
        if return_last_cross_attn:  # 如果请求 cross-attention。 #
            return x, last_cross_attn  # 返回 hidden 和最后一层 cross-attention。 #
        return x  # 返回 decoder hidden。 #


class TahoeStage2MixedModel(nn.Module):  # 定义新的 Stage2 mixed gene-text cross-attention 模型。 #
    def __init__(  # 初始化模型。 #
        self,  # 实例本身。 #
        global_vocab_size: Optional[int] = None,  # global vocab size。 #
        vocab_size: Optional[int] = None,  # vocab size 兼容别名。 #
        d_model: int = 512,  # 隐藏维度。 #
        n_heads: int = 8,  # encoder attention heads。 #
        n_layers: int = 12,  # encoder 层数。 #
        decoder_n_layers: Optional[int] = None,  # decoder 层数。 #
        text_decoder_layers: Optional[int] = None,  # decoder 层数兼容旧配置。 #
        decoder_n_heads: Optional[int] = None,  # decoder heads。 #
        text_decoder_heads: Optional[int] = None,  # decoder heads 兼容旧配置。 #
        expansion_ratio: int = 4,  # FFN 扩张倍率。 #
        dropout: float = 0.1,  # dropout。 #
        pad_token_id: int = 0,  # pad token id。 #
        mask_value: float = -3.0,  # expression mask value。 #
        max_decoder_length: int = 128,  # decoder 最大长度。 #
        value_hidden_dim: int = 128,  # value encoder 隐藏层。 #
        value_head_hidden_dim: int = 256,  # expression head 隐藏层。 #
        tie_lm_head: bool = True,  # 是否绑定 LM head 和 shared embedding 权重。 #
        use_mask_flag_embedding: bool = True,  # 是否使用 mask flag embedding。 #
        **unused_kwargs: Any,  # 吸收旧配置中暂时不用的字段。 #
    ) -> None:  # 初始化结束。 #
        super().__init__()  # 调用父类初始化。 #
        if global_vocab_size is None:  # 如果没有显式传 global vocab size。 #
            global_vocab_size = vocab_size  # 使用 vocab_size 兼容别名。 #
        if global_vocab_size is None:  # 如果仍然没有 vocab size。 #
            raise ValueError("必须传入 global_vocab_size 或 vocab_size。")  # 抛出错误。 #
        self.global_vocab_size = int(global_vocab_size)  # 保存 global vocab size。 #
        self.d_model = int(d_model)  # 保存隐藏维度。 #
        self.pad_token_id = int(pad_token_id)  # 保存 pad token id。 #
        self.mask_value = float(mask_value)  # 保存 mask value。 #
        self.max_decoder_length = int(max_decoder_length)  # 保存 decoder 最大长度。 #
        self.use_mask_flag_embedding = bool(use_mask_flag_embedding)  # 保存 mask flag 开关。 #
        decoder_layers = int(decoder_n_layers if decoder_n_layers is not None else (text_decoder_layers if text_decoder_layers is not None else 2))  # 确定 decoder 层数。 #
        decoder_heads = int(decoder_n_heads if decoder_n_heads is not None else (text_decoder_heads if text_decoder_heads is not None else n_heads))  # 确定 decoder heads。 #
        self.shared_token_embedding = nn.Embedding(self.global_vocab_size, self.d_model, padding_idx=self.pad_token_id)  # 定义 encoder/decoder 共用 token embedding。 #
        self.token_norm = nn.LayerNorm(self.d_model)  # 定义 token embedding LayerNorm。 #
        self.value_encoder = ContinuousValueEncoder(d_model=self.d_model, hidden_dim=value_hidden_dim, dropout=dropout)  # 定义 expression value encoder。 #
        self.mask_flag_embedding = nn.Embedding(2, self.d_model) if self.use_mask_flag_embedding else None  # 定义 mask flag embedding。 #
        self.encoder = DenseTXEncoder(d_model=self.d_model, n_heads=n_heads, n_layers=n_layers, expansion_ratio=expansion_ratio, dropout=dropout)  # 定义 dense encoder。 #
        self.value_head = MaskedValueHead(d_model=self.d_model, hidden_dim=value_head_hidden_dim, dropout=dropout)  # 定义 expression reconstruction head。 #

        # Regulon-specific decoder branch. Encoder-side modules above remain unchanged.
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

        # Annotation-specific decoder branch.
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

        # Keep the original unified vocabulary projection unchanged for optional encoder gene prediction.
        self.unified_lm_head = nn.Linear(self.d_model, self.global_vocab_size, bias=False)
        if bool(tie_lm_head):
            self.unified_lm_head.weight = self.shared_token_embedding.weight

        # Task-specific output heads. They are intentionally independent so that
        # regulon generation and annotation generation do not share decoder output parameters.
        self.regulon_lm_head = nn.Linear(self.d_model, self.global_vocab_size, bias=False)
        self.annotation_lm_head = nn.Linear(self.d_model, self.global_vocab_size, bias=False)
        self._reset_parameters()  # 初始化参数。 #

    def _reset_parameters(self) -> None:  # 初始化模型参数。 #
        nn.init.normal_(self.shared_token_embedding.weight, mean=0.0, std=0.02)  # 初始化 shared embedding。 #
        if self.pad_token_id is not None and 0 <= self.pad_token_id < self.shared_token_embedding.weight.shape[0]:  # 如果 pad id 合法。 #
            with torch.no_grad():  # 不计算梯度。 #
                self.shared_token_embedding.weight[self.pad_token_id].zero_()  # 将 pad embedding 置零。 #
        nn.init.normal_(self.regulon_decoder_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.annotation_decoder_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.regulon_lm_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.annotation_lm_head.weight, mean=0.0, std=0.02)
        if self.mask_flag_embedding is not None:  # 如果使用 mask flag embedding。 #
            nn.init.normal_(self.mask_flag_embedding.weight, mean=0.0, std=0.02)  # 初始化 mask flag embedding。 #

    def encode(self, encoder_input_gene_ids: torch.Tensor, encoder_input_values: torch.Tensor, encoder_key_padding_mask: Optional[torch.Tensor] = None, return_last_attn: bool = False):  # encoder 前向。 #
        token_emb = self.shared_token_embedding(encoder_input_gene_ids.long())  # 使用 shared embedding 编码 gene ids。 #
        token_emb = self.token_norm(token_emb)  # 对 token embedding 做 LayerNorm。 #
        value_emb = self.value_encoder(encoder_input_values.float())  # 编码表达值。 #
        x = token_emb + value_emb  # 融合 gene token embedding 和 expression embedding。 #
        if self.mask_flag_embedding is not None:  # 如果使用 mask flag embedding。 #
            mask_flags = encoder_input_values.eq(self.mask_value).long().clamp(min=0, max=1)  # 根据 mask_value 生成 mask flag。 #
            x = x + self.mask_flag_embedding(mask_flags)  # 加入 mask flag embedding。 #
        if return_last_attn:  # 如果需要 encoder attention。 #
            encoder_outputs, last_attn = self.encoder(x, key_padding_mask=encoder_key_padding_mask, return_last_attn=True)  # encoder 前向并返回 attention。 #
        else:  # 如果不需要 attention。 #
            encoder_outputs = self.encoder(x, key_padding_mask=encoder_key_padding_mask, return_last_attn=False)  # encoder 前向。 #
            last_attn = None  # attention 置空。 #
        return encoder_outputs, last_attn  # 返回 encoder outputs 和可选 attention。 #

    def pool_cell_embedding(self, encoder_outputs: torch.Tensor, encoder_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # 池化得到 cell embedding。 #
        if encoder_outputs.shape[1] > 0:  # 如果序列非空。 #
            return encoder_outputs[:, 0, :]  # 默认使用 <cls> 位置作为 cell embedding。 #
        if encoder_key_padding_mask is None:  # 如果没有 padding mask。 #
            return encoder_outputs.mean(dim=1)  # 返回均值池化。 #
        valid = (~encoder_key_padding_mask).float().unsqueeze(-1)  # 构建有效位置权重。 #
        denom = valid.sum(dim=1).clamp_min(1.0)  # 计算有效位置数量。 #
        return (encoder_outputs * valid).sum(dim=1) / denom  # 返回 masked mean pooling。 #

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

        # Decoder input still uses the original shared global token embedding.
        # This preserves the unified gene-text vocabulary and leaves the encoder embedding parameter unchanged.
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

    def forward(  # 定义完整前向传播。 #
        self,  # 实例本身。 #
        encoder_input_gene_ids: Optional[torch.Tensor] = None,  # encoder 输入 gene ids。 #
        encoder_input_values: Optional[torch.Tensor] = None,  # encoder 输入表达值。 #
        encoder_key_padding_mask: Optional[torch.Tensor] = None,  # encoder padding mask，True 表示 pad。 #
        decoder_input_ids: Optional[torch.Tensor] = None,  # decoder 输入 ids。 #
        decoder_attention_mask: Optional[torch.Tensor] = None,  # decoder attention mask，1 表示有效。 #
        decoder_task_ids: Optional[torch.Tensor] = None,  # 0=regulon，1=annotation；用于 batch 内任务路由。 #
        genes: Optional[torch.Tensor] = None,  # 兼容旧字段 genes。 #
        input_values: Optional[torch.Tensor] = None,  # 兼容旧字段 input_values。 #
        text_input_ids: Optional[torch.Tensor] = None,  # 兼容旧字段 text_input_ids。 #
        text_attention_mask: Optional[torch.Tensor] = None,  # 兼容旧字段 text_attention_mask。 #
        return_encoder_gene_logits: bool = False,  # 是否返回 encoder gene logits。 #
        return_last_attn: bool = False,  # 是否返回 encoder attention。 #
        return_last_cross_attn: bool = False,  # 是否返回 decoder cross-attention。 #
    ) -> Dict[str, torch.Tensor]:  # 返回输出字典。 #
        if encoder_input_gene_ids is None:  # 如果没有新字段 gene ids。 #
            encoder_input_gene_ids = genes  # 使用旧字段 genes。 #
        if encoder_input_values is None:  # 如果没有新字段 input values。 #
            encoder_input_values = input_values  # 使用旧字段 input_values。 #
        if decoder_input_ids is None:  # 如果没有新字段 decoder ids。 #
            decoder_input_ids = text_input_ids  # 使用旧字段 text_input_ids。 #
        if decoder_attention_mask is None:  # 如果没有新字段 decoder mask。 #
            decoder_attention_mask = text_attention_mask  # 使用旧字段 text_attention_mask。 #
        if encoder_input_gene_ids is None or encoder_input_values is None:  # 如果 encoder 必要输入缺失。 #
            raise ValueError("必须提供 encoder_input_gene_ids/encoder_input_values 或 genes/input_values。")  # 抛出错误。 #
        if encoder_key_padding_mask is None:  # 如果没有传 encoder padding mask。 #
            encoder_key_padding_mask = encoder_input_gene_ids.eq(self.pad_token_id)  # 根据 pad token 自动构造。 #
        encoder_outputs, encoder_last_attn = self.encode(encoder_input_gene_ids=encoder_input_gene_ids, encoder_input_values=encoder_input_values, encoder_key_padding_mask=encoder_key_padding_mask, return_last_attn=return_last_attn)  # encoder 前向。 #
        expr_preds = self.value_head(encoder_outputs)  # 预测 expression values。 #
        cell_emb = self.pool_cell_embedding(encoder_outputs, encoder_key_padding_mask=encoder_key_padding_mask)  # 得到 cell embedding。 #
        gene_logits = self.unified_lm_head(encoder_outputs) if return_encoder_gene_logits else None  # 用 unified LM head 预测 masked gene。 #
        # Task-specific decoder outputs. Encoder-side forward logic above is intentionally unchanged.
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

            # Route regulon samples to the regulon-specific decoder.
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

            # Route annotation samples to the annotation-specific decoder.
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




def freeze_encoder_backbone(model: TahoeStage2MixedModel, freeze_embeddings: bool = False, freeze_value_head: bool = False) -> None:  # 冻结 encoder 主干。 #
    for param in model.encoder.parameters():  # 遍历 encoder 参数。 #
        param.requires_grad = False  # 冻结 encoder。 #
    for param in model.value_encoder.parameters():  # 遍历 value encoder 参数。 #
        param.requires_grad = False  # 冻结 value encoder。 #
    if model.mask_flag_embedding is not None:  # 如果存在 mask flag embedding。 #
        for param in model.mask_flag_embedding.parameters():  # 遍历 mask flag embedding 参数。 #
            param.requires_grad = False  # 冻结 mask flag embedding。 #
    if freeze_embeddings:  # 如果要求冻结 shared embedding。 #
        for param in model.shared_token_embedding.parameters():  # 遍历 shared embedding 参数。 #
            param.requires_grad = False  # 冻结 shared embedding。 #
        for param in model.token_norm.parameters():  # 遍历 token norm 参数。 #
            param.requires_grad = False  # 冻结 token norm。 #
    if freeze_value_head:  # 如果要求冻结 expression head。 #
        for param in model.value_head.parameters():  # 遍历 value head 参数。 #
            param.requires_grad = False  # 冻结 value head。 #


TahoeStage2PromptModel = TahoeStage2MixedModel  # 提供旧类名别名，方便旧训练脚本临时兼容。 #

# ========================= Vocab-aware Stage1 loader override ========================= #
# 说明：下面这个函数会覆盖前面同名的 load_stage1_encoder_weights。 #
# 目的：1）继续支持 attn -> self_attn 参数名映射；2）禁止按行号直接拷贝 embedding；3）按 gene_table/global_vocab 精确拷贝 gene embedding。 #

from pathlib import Path as _PathForStage1Loader  # 追加导入 Path，避免依赖文件顶部导入。 #
import os as _os_for_stage1_loader  # 追加导入 os，用于读取环境变量。 #
import json as _json_for_stage1_loader  # 追加导入 json，用于读取 vocab/gene_table。 #
import yaml as _yaml_for_stage1_loader  # 追加导入 yaml，用于读取 stage2_mixed.yaml。 #
import torch as _torch_for_stage1_loader  # 追加导入 torch，用于 checkpoint 和 tensor 操作。 #


def _read_json_or_jsonl_for_stage1_loader(path):  # 读取 json 或 jsonl 文件。 #
    path = _PathForStage1Loader(path)  # 转成 Path。 #
    text = path.read_text(encoding="utf-8")  # 读取文本。 #
    if path.suffix.lower() == ".jsonl":  # 如果是 jsonl。 #
        return [_json_for_stage1_loader.loads(line) for line in text.splitlines() if line.strip()]  # 逐行解析。 #
    try:  # 尝试按普通 json 解析。 #
        return _json_for_stage1_loader.loads(text)  # 返回 json 对象。 #
    except Exception:  # 如果普通 json 失败。 #
        return [_json_for_stage1_loader.loads(line) for line in text.splitlines() if line.strip()]  # 回退为 jsonl。 #


def _build_token_to_id_for_stage1_loader(vocab_obj):  # 构建 token -> id 映射。 #
    if isinstance(vocab_obj, dict) and "token_to_id" in vocab_obj:  # 如果已有 token_to_id。 #
        return {str(k): int(v) for k, v in vocab_obj["token_to_id"].items()}  # 返回映射。 #
    if isinstance(vocab_obj, dict) and "id_to_token" in vocab_obj:  # 如果是 id_to_token。 #
        return {str(v): int(k) for k, v in vocab_obj["id_to_token"].items()}  # 反转为 token_to_id。 #
    if isinstance(vocab_obj, dict):  # 如果是普通 dict。 #
        if all(isinstance(v, int) for v in vocab_obj.values()):  # 判断 token -> id。 #
            return {str(k): int(v) for k, v in vocab_obj.items()}  # 返回 token_to_id。 #
        if all(str(k).isdigit() for k in vocab_obj.keys()):  # 判断 id -> token。 #
            return {str(v): int(k) for k, v in vocab_obj.items()}  # 反转为 token_to_id。 #
    if isinstance(vocab_obj, list):  # 如果是 list[dict]。 #
        out = {}  # 初始化映射。 #
        for item in vocab_obj:  # 遍历记录。 #
            if not isinstance(item, dict):  # 跳过非 dict。 #
                continue  # 继续。 #
            tid = item.get("token_id", item.get("id", item.get("global_id", None)))  # 获取 id。 #
            tok = item.get("global_token", item.get("token", item.get("gene_symbol", item.get("ensembl_id", None))))  # 获取 token。 #
            if tid is not None and tok is not None:  # 如果二者都有。 #
                out[str(tok)] = int(tid)  # 写入 token_to_id。 #
        return out  # 返回映射。 #
    return {}  # 解析失败则返回空。 #


def _build_global_token_to_id_for_stage1_loader(global_vocab_path):  # 构建 global token -> global id 映射。 #
    obj = _read_json_or_jsonl_for_stage1_loader(global_vocab_path)  # 读取 global vocab。 #
    return _build_token_to_id_for_stage1_loader(obj)  # 转为 token_to_id。 #


def _find_stage2_yaml_for_stage1_loader():  # 寻找当前 stage2 yaml。 #
    candidates = [  # 候选配置路径。 #
        _os_for_stage1_loader.environ.get("STAGE2_CONFIG_PATH", ""),  # 环境变量指定路径。 #
        "stage2.yaml",  # 当前项目默认配置。 #
        "stage2_mixed.yaml",  # 兼容旧配置文件名。 #
        "/root/autodl-tmp/ExpertCoder/ExpertCoder_2A_CA/stage2_mixed.yaml",  # 绝对路径。 #
    ]  # 候选结束。 #
    for p in candidates:  # 遍历候选。 #
        if p and _PathForStage1Loader(p).exists():  # 如果存在。 #
            return _PathForStage1Loader(p)  # 返回路径。 #
    return None  # 没找到返回 None。 #


def _get_vocab_paths_for_stage1_loader():  # 从环境变量或 yaml 里读取 vocab 路径。 #
    yaml_path = _find_stage2_yaml_for_stage1_loader()  # 寻找 yaml。 #
    cfg = {}  # 初始化配置。 #
    if yaml_path is not None:  # 如果找到 yaml。 #
        cfg = _yaml_for_stage1_loader.safe_load(yaml_path.read_text(encoding="utf-8")) or {}  # 读取 yaml。 #
    gv_cfg = cfg.get("global_vocab", {}) if isinstance(cfg, dict) else {}  # 读取 global_vocab 配置。 #

    global_vocab_path = _os_for_stage1_loader.environ.get("GLOBAL_VOCAB_PATH") or gv_cfg.get("global_vocab_path")  # global vocab 路径。 #
    gene_table_path = _os_for_stage1_loader.environ.get("GENE_TABLE_PATH") or gv_cfg.get("gene_table_path")  # gene table 路径。 #
    stage1_vocab_path = (  # Stage1 旧 vocab 路径，可选。 #
        _os_for_stage1_loader.environ.get("STAGE1_VOCAB_PATH")  # 环境变量。 #
        or gv_cfg.get("stage1_vocab_path")  # 推荐字段。 #
        or gv_cfg.get("raw_gene_vocab_path")  # 兼容字段。 #
        or gv_cfg.get("old_gene_vocab_path")  # 兼容字段。 #
    )  # 结束。 #
    return global_vocab_path, gene_table_path, stage1_vocab_path  # 返回三个路径。 #


def _select_stage1_embedding_key_for_stage1_loader(state, own_state):  # 自动寻找 Stage1 gene embedding 参数名。 #
    candidate_keys = [  # 常见 embedding key。 #
        "gene_encoder.embedding.weight",  # 当前日志中出现过的 key。 #
        "gene_embedding.weight",  # 兼容 key。 #
        "token_embedding.weight",  # 兼容 key。 #
        "gene_token_embedding.weight",  # 兼容 key。 #
        "embedding.weight",  # 兼容 key。 #
        "shared_token_embedding.weight",  # 兼容 key。 #
    ]  # 候选结束。 #
    for k in candidate_keys:  # 遍历候选。 #
        if k in state and "shared_token_embedding.weight" in own_state:  # 如果 checkpoint 和模型都有对应项。 #
            if state[k].ndim == 2 and state[k].shape[1] == own_state["shared_token_embedding.weight"].shape[1]:  # 检查维度。 #
                return k  # 返回 key。 #
    for k, v in state.items():  # 如果候选没找到，遍历所有参数。 #
        lk = k.lower()  # 小写 key。 #
        if "embedding" in lk and hasattr(v, "shape") and v.ndim == 2:  # 找二维 embedding。 #
            if "shared_token_embedding.weight" in own_state and v.shape[1] == own_state["shared_token_embedding.weight"].shape[1]:  # 检查维度。 #
                return k  # 返回 key。 #
    return None  # 未找到返回 None。 #


def _copy_embedding_by_gene_table_for_stage1_loader(model, state, copied_info, verbose=True):  # 按 gene_table 精确拷贝 gene embedding。 #
    own_state = model.state_dict()  # 获取当前模型 state_dict。 #
    if "shared_token_embedding.weight" not in own_state:  # 如果模型没有 shared embedding。 #
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": "no shared_token_embedding.weight"}  # 记录原因。 #
        return  # 返回。 #

    global_vocab_path, gene_table_path, stage1_vocab_path = _get_vocab_paths_for_stage1_loader()  # 获取路径。 #
    if global_vocab_path is None or gene_table_path is None:  # 如果缺必要路径。 #
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": "missing global_vocab_path or gene_table_path"}  # 记录原因。 #
        return  # 返回。 #

    global_vocab_path = _PathForStage1Loader(global_vocab_path)  # 转 Path。 #
    gene_table_path = _PathForStage1Loader(gene_table_path)  # 转 Path。 #
    if not global_vocab_path.exists() or not gene_table_path.exists():  # 如果路径不存在。 #
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": f"path not exists: {global_vocab_path}, {gene_table_path}"}  # 记录原因。 #
        return  # 返回。 #

    global_token_to_id = _build_global_token_to_id_for_stage1_loader(global_vocab_path)  # 构建 global token -> id。 #
    gene_table = _read_json_or_jsonl_for_stage1_loader(gene_table_path)  # 读取 gene table。 #
    if isinstance(gene_table, dict):  # 如果 gene table 是 dict。 #
        gene_table = gene_table.get("genes", gene_table.get("items", gene_table.get("data", [])))  # 尝试取列表。 #

    old_token_to_id = {}  # 初始化旧 vocab token -> id。 #
    if stage1_vocab_path is not None and _PathForStage1Loader(stage1_vocab_path).exists():  # 如果旧 vocab 路径存在。 #
        old_token_to_id = _build_token_to_id_for_stage1_loader(_read_json_or_jsonl_for_stage1_loader(stage1_vocab_path))  # 读取旧 vocab。 #

    old_embed_key = _select_stage1_embedding_key_for_stage1_loader(state, own_state)  # 找旧 embedding key。 #
    if old_embed_key is None:  # 如果没找到。 #
        copied_info["embedding_by_gene_table"] = {"copied": 0, "reason": "no stage1 embedding key found"}  # 记录原因。 #
        return  # 返回。 #

    old_embed = state[old_embed_key]  # 旧 embedding。 #
    new_embed = own_state["shared_token_embedding.weight"]  # 新 shared embedding。 #
    copied = 0  # 拷贝计数。 #
    skipped = 0  # 跳过计数。 #
    examples = []  # 示例。 #

    with _torch_for_stage1_loader.no_grad():  # 不记录梯度。 #
        for item in gene_table:  # 遍历 gene table。 #
            if not isinstance(item, dict):  # 跳过异常记录。 #
                skipped += 1  # 计数。 #
                continue  # 继续。 #

            old_id = item.get("token_id", item.get("stage1_token_id", item.get("old_token_id", None)))  # 优先从 gene_table 取旧 id。 #
            gene_symbol = item.get("gene_symbol", item.get("symbol", None))  # 读取 gene symbol。 #
            ensembl_id = item.get("ensembl_id", item.get("ensembl", None))  # 读取 Ensembl id。 #
            global_id = item.get("global_id", item.get("global_token_id", item.get("new_token_id", None)))  # 尝试读取 global id。 #
            global_token = item.get("global_token", None)  # 尝试读取 global token。 #

            if old_id is None and old_token_to_id:  # 如果 gene_table 没有 old_id，则尝试用旧 vocab 查。 #
                for tok in [gene_symbol, ensembl_id, f"<gene:{gene_symbol}>" if gene_symbol else None, f"<gene:{ensembl_id}>" if ensembl_id else None]:  # 构建候选旧 token。 #
                    if tok is not None and str(tok) in old_token_to_id:  # 如果旧 vocab 有。 #
                        old_id = old_token_to_id[str(tok)]  # 得到 old_id。 #
                        break  # 停止。 #

            if global_id is None:  # 如果没有 global id。 #
                candidates = []  # 初始化候选 global token。 #
                if global_token is not None:  # 如果 gene_table 已有 global_token。 #
                    candidates.append(str(global_token))  # 加入候选。 #
                if gene_symbol is not None:  # 如果有 symbol。 #
                    candidates.append(f"<gene:{gene_symbol}>")  # 加入 symbol gene token。 #
                if ensembl_id is not None:  # 如果有 Ensembl。 #
                    candidates.append(f"<gene:{ensembl_id}>")  # 加入 Ensembl gene token。 #
                for tok in candidates:  # 遍历候选。 #
                    if tok in global_token_to_id:  # 如果 global vocab 里存在。 #
                        global_id = global_token_to_id[tok]  # 得到 global id。 #
                        global_token = tok  # 记录 token。 #
                        break  # 停止。 #

            if old_id is None or global_id is None:  # 如果 id 不完整。 #
                skipped += 1  # 跳过计数。 #
                continue  # 继续。 #
            old_id = int(old_id)  # 转 int。 #
            global_id = int(global_id)  # 转 int。 #
            if old_id < 0 or old_id >= old_embed.shape[0] or global_id < 0 or global_id >= new_embed.shape[0]:  # 检查范围。 #
                skipped += 1  # 跳过。 #
                continue  # 继续。 #

            new_embed[global_id].copy_(old_embed[old_id])  # 按 id 精确拷贝 embedding。 #
            copied += 1  # 计数。 #
            if len(examples) < 10:  # 保存前 10 个例子。 #
                examples.append({"old_id": old_id, "global_id": global_id, "global_token": global_token, "gene_symbol": gene_symbol, "ensembl_id": ensembl_id})  # 加入示例。 #

    copied_info["embedding_by_gene_table"] = {  # 记录结果。 #
        "copied": copied,  # 成功拷贝数量。 #
        "skipped": skipped,  # 跳过数量。 #
        "old_embed_key": old_embed_key,  # 旧 embedding key。 #
        "global_vocab_path": str(global_vocab_path),  # global vocab 路径。 #
        "gene_table_path": str(gene_table_path),  # gene table 路径。 #
        "stage1_vocab_path": str(stage1_vocab_path) if stage1_vocab_path else None,  # 旧 vocab 路径。 #
        "examples": examples,  # 示例。 #
    }  # 记录结束。 #
    if verbose:  # 如果打印日志。 #
        print(f"[vocab-aware embedding copy] copied={copied}, skipped={skipped}, old_embed_key={old_embed_key}")  # 打印概况。 #
        print("[vocab-aware examples]", examples[:5])  # 打印示例。 #


def _copy_special_embeddings_by_exact_token_for_stage1_loader(model, state, copied_info, verbose=True):  # 可选：按同名 token 拷贝 <cls>/<pad> 等特殊 token。 #
    own_state = model.state_dict()  # 获取模型 state_dict。 #
    global_vocab_path, gene_table_path, stage1_vocab_path = _get_vocab_paths_for_stage1_loader()  # 获取路径。 #
    if stage1_vocab_path is None or global_vocab_path is None:  # 如果缺路径。 #
        copied_info["special_embedding_by_name"] = {"copied": 0, "reason": "missing stage1_vocab_path or global_vocab_path"}  # 记录原因。 #
        return  # 返回。 #
    if not _PathForStage1Loader(stage1_vocab_path).exists() or not _PathForStage1Loader(global_vocab_path).exists():  # 如果不存在。 #
        copied_info["special_embedding_by_name"] = {"copied": 0, "reason": "path not exists"}  # 记录原因。 #
        return  # 返回。 #

    old_token_to_id = _build_token_to_id_for_stage1_loader(_read_json_or_jsonl_for_stage1_loader(stage1_vocab_path))  # 旧 token_to_id。 #
    global_token_to_id = _build_global_token_to_id_for_stage1_loader(global_vocab_path)  # 新 token_to_id。 #
    old_embed_key = _select_stage1_embedding_key_for_stage1_loader(state, own_state)  # 找旧 embedding key。 #
    if old_embed_key is None or "shared_token_embedding.weight" not in own_state:  # 如果缺 key。 #
        copied_info["special_embedding_by_name"] = {"copied": 0, "reason": "missing embedding"}  # 记录原因。 #
        return  # 返回。 #

    old_embed = state[old_embed_key]  # 旧 embedding。 #
    new_embed = own_state["shared_token_embedding.weight"]  # 新 embedding。 #
    allowed_specials = ["<cls>", "<pad>"]  # 只拷贝语义明确的特殊 token，不拷贝 <eoc>/<junk*>。 #
    copied = 0  # 计数。 #
    examples = []  # 示例。 #
    with _torch_for_stage1_loader.no_grad():  # 不记录梯度。 #
        for tok in allowed_specials:  # 遍历特殊 token。 #
            if tok in old_token_to_id and tok in global_token_to_id:  # 如果新旧都有该 token。 #
                old_id = int(old_token_to_id[tok])  # 旧 id。 #
                new_id = int(global_token_to_id[tok])  # 新 id。 #
                if old_id < old_embed.shape[0] and new_id < new_embed.shape[0]:  # 检查范围。 #
                    new_embed[new_id].copy_(old_embed[old_id])  # 精确拷贝。 #
                    copied += 1  # 计数。 #
                    examples.append({"token": tok, "old_id": old_id, "new_id": new_id})  # 记录示例。 #
    copied_info["special_embedding_by_name"] = {"copied": copied, "examples": examples}  # 记录结果。 #
    if verbose and copied > 0:  # 如果需要打印。 #
        print(f"[special embedding copy] copied={copied}, examples={examples}")  # 打印。 #


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

    # 1. 优先直接加载 embedding
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

    # 2. 加载其余 Stage1 参数
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

    # 3. embedding shape 不一致时再回退到旧映射
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
