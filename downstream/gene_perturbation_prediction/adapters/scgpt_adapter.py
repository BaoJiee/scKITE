import copy
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseSCFMAdapter


class _GeneEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(
            int(vocab_size), int(embedding_dim), padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(int(embedding_dim))

    def forward(self, token_ids):
        return self.enc_norm(self.embedding(token_ids))


class _ContinuousValueEncoder(nn.Module):
    def __init__(self, embedding_dim, dropout=0.0, max_value=512):
        super().__init__()
        embedding_dim = int(embedding_dim)
        self.linear1 = nn.Linear(1, embedding_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.max_value = float(max_value)

    def forward(self, values):
        values = torch.clamp(values.unsqueeze(-1), max=self.max_value)
        values = self.activation(self.linear1(values))
        values = self.linear2(values)
        values = self.norm(values)
        return self.dropout(values)


class _FrozenScGPTEncoder(nn.Module):
    """Minimal scGPT encoder with checkpoint-compatible parameter names."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        feedforward_dim,
        num_layers,
        dropout,
        pad_token_id,
        pre_norm=False,
    ):
        super().__init__()
        self.encoder = _GeneEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )
        self.value_encoder = _ContinuousValueEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(embedding_dim),
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=bool(pre_norm),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )

    def forward(self, token_ids, values, key_padding_mask):
        gene_embeddings = self.encoder(token_ids)
        value_embeddings = self.value_encoder(values)
        return self.transformer_encoder(
            gene_embeddings + value_embeddings,
            src_key_padding_mask=key_padding_mask,
        )


class ScGPTAdapter(BaseSCFMAdapter):
    """Use a frozen scGPT checkpoint as the embedding provider for GEARS.

    This adapter intentionally does not import the ``scgpt`` package.  It avoids
    the torch/torchtext ABI dependency and reconstructs only the encoder modules
    needed by GEARS.  The GEARS model, loss, and training workflow remain intact.
    """

    name = "scgpt"

    def __init__(
        self,
        model_dir=None,
        ckpt_path=None,
        vocab_path=None,
        args_path=None,
        scgpt_source_dir=None,
        device="cuda",
        gears_hidden_size=512,
        max_seq_len=None,
        num_bins=None,
        contextual_value_mode="bin",
        contextual_gene_selection="sample",
        contextual_fallback="static",
        contextual_encoder_batch_size=None,
        missing_strategy="mean_gene",
        project_method="slice",
        normalize=False,
        precision="fp16",
        seed=1,
        allow_flag_encoder=True,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        model_dir = Path(model_dir) if model_dir is not None else None
        self.model_dir = model_dir
        if model_dir is None and not all((ckpt_path, vocab_path, args_path)):
            raise ValueError(
                "Provide model_dir, or provide ckpt_path, vocab_path, and args_path."
            )
        self.ckpt_path = (
            Path(ckpt_path) if ckpt_path else model_dir / "best_model.pt"
        )
        self.vocab_path = Path(vocab_path) if vocab_path else model_dir / "vocab.json"
        self.args_path = Path(args_path) if args_path else model_dir / "args.json"
        self.scgpt_source_dir = (
            Path(scgpt_source_dir) if scgpt_source_dir is not None else None
        )

        for label, path in (
            ("checkpoint", self.ckpt_path),
            ("vocabulary", self.vocab_path),
            ("model args", self.args_path),
        ):
            if path is None or not path.exists():
                raise FileNotFoundError(f"scGPT {label} file not found: {path}")

        self.device = str(device)
        self.gears_hidden_size = int(gears_hidden_size)
        self.output_dim = self.gears_hidden_size
        self.contextual_value_mode = str(contextual_value_mode).lower()
        self.contextual_gene_selection = str(contextual_gene_selection).lower()
        self.contextual_fallback = str(contextual_fallback).lower()
        self.contextual_encoder_batch_size = (
            None
            if contextual_encoder_batch_size is None
            or int(contextual_encoder_batch_size) <= 0
            else int(contextual_encoder_batch_size)
        )
        self.missing_strategy = str(missing_strategy).lower()
        self.project_method = str(project_method).lower()
        self.normalize = bool(normalize)
        self.precision = str(precision).lower()
        self.seed = int(seed)
        self.allow_flag_encoder = bool(allow_flag_encoder)

        if self.contextual_value_mode not in {"bin", "as_is", "raw", "none", "log1p"}:
            raise ValueError(
                "contextual_value_mode must be one of: bin, as_is, raw, none, log1p"
            )
        if self.contextual_gene_selection not in {
            "sample",
            "top_expression",
            "first",
            "matched_first",
        }:
            raise ValueError(
                "contextual_gene_selection must be one of: "
                "sample, top_expression, first, matched_first"
            )
        if self.contextual_fallback not in {"static", "zero"}:
            raise ValueError("contextual_fallback must be 'static' or 'zero'.")
        if self.missing_strategy not in {"mean_gene", "mean_all", "zero"}:
            raise ValueError(
                "missing_strategy must be one of: mean_gene, mean_all, zero"
            )
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16")

        self.model_args = json.loads(self.args_path.read_text(encoding="utf-8"))
        self.raw_output_dim = int(self.model_args["embsize"])
        self.max_seq_len = int(
            max_seq_len
            if max_seq_len is not None and int(max_seq_len) > 0
            else self.model_args.get("max_seq_len", 1200)
        )
        self.num_bins = int(
            num_bins
            if num_bins is not None and int(num_bins) > 1
            else self.model_args.get("n_bins", 51)
        )
        self.pad_value = float(self.model_args.get("pad_value", -2))
        self.pad_token = str(self.model_args.get("pad_token", "<pad>"))
        self.use_cls = not bool(
            self.model_args.get(
                "no_cls",
                not bool(self.model_args.get("USE_CLS", False)),
            )
        )

        input_emb_style = str(
            self.model_args.get("input_emb_style", "continuous")
        ).lower()
        if input_emb_style != "continuous":
            raise ValueError(
                "This adapter currently supports scGPT input_emb_style='continuous' "
                f"only, but args.json contains {input_emb_style!r}."
            )

        self.token_to_id = self._load_vocab(self.vocab_path)
        if self.pad_token not in self.token_to_id:
            raise KeyError(
                f"pad token {self.pad_token!r} is missing from {self.vocab_path}"
            )
        self.pad_token_id = int(self.token_to_id[self.pad_token])
        self.cls_token_id = self.token_to_id.get("<cls>")
        if self.use_cls and self.cls_token_id is None:
            raise KeyError("args.json enables CLS but vocab.json has no '<cls>' token.")

        self._upper_token_to_id = {}
        for token, token_id in self.token_to_id.items():
            if token.startswith("<") and token.endswith(">"):
                continue
            self._upper_token_to_id.setdefault(token.upper(), int(token_id))

        self._gene_embedding_weight, self.flag_embedding_shape = (
            self._load_static_checkpoint_tensors()
        )
        if self._gene_embedding_weight.shape[0] != len(self.token_to_id):
            max_vocab_id = max(self.token_to_id.values())
            if max_vocab_id >= self._gene_embedding_weight.shape[0]:
                raise ValueError(
                    "vocab token ids exceed encoder.embedding.weight rows: "
                    f"max_vocab_id={max_vocab_id}, rows={self._gene_embedding_weight.shape[0]}"
                )
        if self._gene_embedding_weight.shape[1] != self.raw_output_dim:
            raise ValueError(
                "args.json embsize does not match checkpoint gene embedding: "
                f"{self.raw_output_dim} vs {self._gene_embedding_weight.shape[1]}"
            )

        self.gene_list = None
        self.gene_id_list = None
        self.pert_list = None
        self.gene_token_ids = None
        self.pert_token_ids = None
        self._gene_token_tensor = None
        self._matched_gene_indices = None
        self._missing_fill_cache = None
        self._static_gene_embeddings_cache = None
        self._static_gene_embeddings_device_cache = None
        self.scgpt_encoder = None
        self.loaded_encoder_key_count = 0
        self._np_rng = np.random.default_rng(self.seed)
        self._torch_generator = torch.Generator(device="cpu")
        self._torch_generator.manual_seed(self.seed)

        print(
            "[ScGPTAdapter] "
            f"vocab={len(self.token_to_id)}, raw_dim={self.raw_output_dim}, "
            f"output_dim={self.output_dim}, layers={self.model_args.get('nlayers')}, "
            f"heads={self.model_args.get('nheads')}, max_seq_len={self.max_seq_len}, "
            f"use_cls={self.use_cls}, value_mode={self.contextual_value_mode}"
        )
        if self.flag_embedding_shape is not None:
            print(
                "[ScGPTAdapter] checkpoint-only flag_encoder.weight detected "
                f"with shape={self.flag_embedding_shape}; it is not used because "
                "the supplied inference source has no flag_encoder path."
            )

    def __deepcopy__(self, memo):
        """Share immutable frozen weights across GEARS best-model deep copies."""
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        shared = {
            "scgpt_encoder",
            "_gene_embedding_weight",
            "_gene_token_tensor",
            "_matched_gene_indices",
            "_missing_fill_cache",
            "_static_gene_embeddings_cache",
            "_static_gene_embeddings_device_cache",
        }
        for key, value in self.__dict__.items():
            if key in shared:
                setattr(result, key, value)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result

    @staticmethod
    def _load_vocab(path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for nested_key in ("stoi", "token_to_id", "vocab"):
                nested = obj.get(nested_key)
                if isinstance(nested, dict):
                    obj = nested
                    break
            if all(isinstance(v, (int, np.integer)) for v in obj.values()):
                return {str(k): int(v) for k, v in obj.items()}

        if isinstance(obj, list):
            if all(isinstance(item, str) for item in obj):
                return {str(token): idx for idx, token in enumerate(obj)}
            mapping = {}
            for idx, item in enumerate(obj):
                if not isinstance(item, dict):
                    continue
                token = item.get("token", item.get("gene", item.get("gene_symbol")))
                token_id = item.get("id", item.get("token_id", idx))
                if token is not None:
                    mapping[str(token)] = int(token_id)
            if mapping:
                return mapping

        raise ValueError(
            "Unsupported scGPT vocab.json format. Expected token->id mapping or token list."
        )

    @staticmethod
    def _torch_load(path):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Expected a checkpoint mapping, received {type(checkpoint).__name__}."
            )
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
        state = {}
        for key, value in checkpoint.items():
            key = str(key)
            if key.startswith("module."):
                key = key[len("module.") :]
            state[key] = value
        return state

    def _load_static_checkpoint_tensors(self):
        state = self._extract_state_dict(self._torch_load(self.ckpt_path))
        key = "encoder.embedding.weight"
        if key not in state:
            embedding_like = [k for k in state if "embedding" in k.lower()][:20]
            raise KeyError(
                f"{key!r} is missing from scGPT checkpoint. Candidates: {embedding_like}"
            )
        embedding = state[key].detach().float().cpu()
        flag_shape = None
        if "flag_encoder.weight" in state:
            flag_shape = tuple(state["flag_encoder.weight"].shape)
            if not self.allow_flag_encoder:
                raise RuntimeError(
                    "Checkpoint contains flag_encoder.weight but allow_flag_encoder=False."
                )
        return embedding, flag_shape

    @staticmethod
    def _convert_attention_key(key):
        key = re.sub(
            r"self_attn\._impl\.Wqkv\.(weight|bias)$",
            r"self_attn.in_proj_\1",
            key,
        )
        key = re.sub(
            r"self_attn\.Wqkv\.(weight|bias)$",
            r"self_attn.in_proj_\1",
            key,
        )
        key = re.sub(
            r"self_attn\._impl\.out_proj\.",
            "self_attn.out_proj.",
            key,
        )
        return key

    def _ensure_encoder_loaded(self):
        if self.scgpt_encoder is not None:
            return

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"ScGPTAdapter requested device={self.device}, but CUDA is unavailable."
            )

        model = _FrozenScGPTEncoder(
            vocab_size=self._gene_embedding_weight.shape[0],
            embedding_dim=self.raw_output_dim,
            num_heads=int(self.model_args["nheads"]),
            feedforward_dim=int(self.model_args["d_hid"]),
            num_layers=int(self.model_args["nlayers"]),
            dropout=float(self.model_args.get("dropout", 0.0)),
            pad_token_id=self.pad_token_id,
            pre_norm=bool(self.model_args.get("pre_norm", False)),
        )

        raw_state = self._extract_state_dict(self._torch_load(self.ckpt_path))
        converted_state = {
            self._convert_attention_key(key): value
            for key, value in raw_state.items()
            if key.startswith(
                ("encoder.", "value_encoder.", "transformer_encoder.")
            )
        }
        model_state = model.state_dict()
        shape_mismatch = []
        unexpected_encoder_keys = []
        compatible = {}
        for key, value in converted_state.items():
            if key not in model_state:
                unexpected_encoder_keys.append(key)
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                shape_mismatch.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape))
                )
                continue
            compatible[key] = value

        if shape_mismatch:
            details = "\n".join(
                f"  {key}: checkpoint={ckpt_shape}, model={model_shape}"
                for key, ckpt_shape, model_shape in shape_mismatch[:50]
            )
            raise RuntimeError(
                "scGPT encoder parameter shapes do not match args.json:\n" + details
            )

        if unexpected_encoder_keys:
            details = "\n".join(
                f"  {key}" for key in unexpected_encoder_keys[:100]
            )
            raise RuntimeError(
                "Checkpoint contains unrecognized encoder parameters; refusing to "
                "silently skip them:\n" + details
            )

        missing = [key for key in model_state if key not in compatible]
        if missing:
            details = "\n".join(f"  {key}" for key in missing[:100])
            raise RuntimeError(
                "Critical scGPT encoder parameters are missing after Wqkv conversion:\n"
                + details
            )

        model.load_state_dict(compatible, strict=True)

        loaded_state = model.state_dict()
        failed = []
        for key, checkpoint_tensor in compatible.items():
            expected = checkpoint_tensor.detach().cpu().to(loaded_state[key].dtype)
            actual = loaded_state[key].detach().cpu()
            if not torch.equal(expected, actual):
                failed.append(key)
        if failed:
            raise RuntimeError(
                "Exact tensor verification failed for scGPT encoder keys: "
                + ", ".join(failed[:50])
            )

        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.scgpt_encoder = model
        self.loaded_encoder_key_count = len(compatible)
        print(
            "[ScGPTAdapter] encoder loading verification passed: "
            f"{self.loaded_encoder_key_count}/{len(model_state)} tensors loaded exactly."
        )

    @staticmethod
    def _strip_version(name):
        if name is None:
            return None
        name = str(name)
        left, separator, right = name.rpartition(".")
        return left if separator and right.isdigit() else name

    def _lookup_token_id(self, name):
        if name is None:
            return None
        name = str(name)
        candidates = [name, self._strip_version(name), name.upper()]
        for candidate in candidates:
            if candidate in self.token_to_id:
                return int(self.token_to_id[candidate])
        return self._upper_token_to_id.get(name.upper())

    def setup(
        self,
        gene_list: List[str],
        pert_list: List[str],
        gene_id_list: Optional[List[str]] = None,
    ):
        self.gene_list = list(gene_list)
        self.gene_id_list = (
            list(gene_id_list) if gene_id_list is not None else [None] * len(gene_list)
        )
        self.pert_list = list(pert_list)
        if len(self.gene_id_list) != len(self.gene_list):
            raise ValueError(
                "gene_list and gene_id_list length mismatch: "
                f"{len(self.gene_list)} vs {len(self.gene_id_list)}"
            )

        self.gene_token_ids = []
        self.gene_match_sources = []
        for gene_symbol, gene_id in zip(self.gene_list, self.gene_id_list):
            token_id = self._lookup_token_id(gene_symbol)
            source = "gene_symbol"
            if token_id is None:
                token_id = self._lookup_token_id(gene_id)
                source = "var_index" if token_id is not None else "missing"
            self.gene_token_ids.append(token_id)
            self.gene_match_sources.append(source)

        self.pert_token_ids = [self._lookup_token_id(name) for name in self.pert_list]
        gene_matched = sum(token_id is not None for token_id in self.gene_token_ids)
        pert_matched = sum(token_id is not None for token_id in self.pert_token_ids)
        if gene_matched == 0:
            raise ValueError("No GEARS genes matched the scGPT vocabulary.")
        if pert_matched == 0:
            raise ValueError("No GEARS perturbations matched the scGPT vocabulary.")

        token_tensor = [
            self.pad_token_id if token_id is None else int(token_id)
            for token_id in self.gene_token_ids
        ]
        self._gene_token_tensor = torch.tensor(token_tensor, dtype=torch.long)
        self._matched_gene_indices = torch.tensor(
            [
                index
                for index, token_id in enumerate(self.gene_token_ids)
                if token_id is not None
            ],
            dtype=torch.long,
        )
        self._missing_fill_cache = None
        self._static_gene_embeddings_cache = self._project_to_hidden(
            self._embedding_by_token_ids(self.gene_token_ids)
        ).detach()
        self._static_gene_embeddings_device_cache = None

        print(
            "[ScGPTAdapter] GEARS gene coverage: "
            f"{gene_matched}/{len(self.gene_list)} = "
            f"{gene_matched / len(self.gene_list):.4f}"
        )
        print(
            "[ScGPTAdapter] GEARS perturbation coverage: "
            f"{pert_matched}/{len(self.pert_list)} = "
            f"{pert_matched / len(self.pert_list):.4f}"
        )

    def _missing_fill_vector(self):
        if self._missing_fill_cache is not None:
            return self._missing_fill_cache
        if self.missing_strategy == "zero":
            fill = torch.zeros(self.raw_output_dim, dtype=torch.float32)
        elif self.missing_strategy == "mean_all":
            fill = self._gene_embedding_weight.mean(dim=0)
        else:
            gene_ids = [
                token_id
                for token, token_id in self.token_to_id.items()
                if not (token.startswith("<") and token.endswith(">"))
                and 0 <= int(token_id) < self._gene_embedding_weight.shape[0]
            ]
            if not gene_ids:
                fill = self._gene_embedding_weight.mean(dim=0)
            else:
                fill = self._gene_embedding_weight[
                    torch.tensor(gene_ids, dtype=torch.long)
                ].mean(dim=0)
        self._missing_fill_cache = fill.detach()
        return self._missing_fill_cache

    def _embedding_by_token_ids(self, token_ids):
        fill = self._missing_fill_vector()
        max_token_id = self._gene_embedding_weight.shape[0] - 1
        rows = []
        for token_id in token_ids:
            if token_id is None or not 0 <= int(token_id) <= max_token_id:
                rows.append(fill)
            else:
                rows.append(self._gene_embedding_weight[int(token_id)])
        return torch.stack(rows, dim=0)

    def _project_to_hidden(self, embeddings):
        original_shape = embeddings.shape
        embeddings = embeddings.reshape(-1, original_shape[-1])
        if embeddings.shape[1] == self.gears_hidden_size:
            projected = embeddings
        elif self.project_method == "slice":
            if embeddings.shape[1] > self.gears_hidden_size:
                projected = embeddings[:, : self.gears_hidden_size].contiguous()
            else:
                padding = torch.zeros(
                    embeddings.shape[0],
                    self.gears_hidden_size - embeddings.shape[1],
                    dtype=embeddings.dtype,
                    device=embeddings.device,
                )
                projected = torch.cat([embeddings, padding], dim=1)
        elif self.project_method == "mean_pool":
            if embeddings.shape[1] % self.gears_hidden_size != 0:
                raise ValueError(
                    "mean_pool requires raw_output_dim % gears_hidden_size == 0."
                )
            group_size = embeddings.shape[1] // self.gears_hidden_size
            projected = embeddings.reshape(
                embeddings.shape[0], self.gears_hidden_size, group_size
            ).mean(dim=2)
        else:
            raise ValueError(f"Unknown project_method: {self.project_method}")

        if self.normalize:
            projected = torch.nn.functional.normalize(projected, p=2, dim=1)
        return projected.reshape(*original_shape[:-1], self.gears_hidden_size)

    def get_static_gene_embeddings(self, gene_list: List[str]):
        if self.gene_token_ids is None:
            raise RuntimeError("Call setup() before requesting scGPT embeddings.")
        if len(gene_list) == len(self.gene_token_ids):
            return self._static_gene_embeddings_cache.clone()
        token_ids = (
            self.gene_token_ids
            if len(gene_list) == len(self.gene_token_ids)
            else [self._lookup_token_id(name) for name in gene_list]
        )
        return self._project_to_hidden(
            self._embedding_by_token_ids(token_ids)
        ).clone()

    def get_pert_embeddings(self, pert_list: List[str]):
        if self.pert_token_ids is None:
            raise RuntimeError("Call setup() before requesting scGPT embeddings.")
        token_ids = [self._lookup_token_id(name) for name in pert_list]
        return self._project_to_hidden(
            self._embedding_by_token_ids(token_ids)
        ).clone()

    def _bin_values(self, values):
        values_np = values.detach().float().cpu().numpy()
        if values_np.size == 0:
            return torch.empty(0, dtype=torch.float32)
        if values_np.min() < 0:
            raise ValueError("scGPT bin mode requires non-negative expression values.")
        if values_np.max() == 0:
            return torch.zeros(len(values_np), dtype=torch.float32)

        nonzero_mask = values_np != 0
        nonzero_values = values_np[nonzero_mask]
        bins = np.quantile(
            nonzero_values,
            np.linspace(0, 1, self.num_bins - 1),
        )
        left = np.digitize(nonzero_values, bins)
        right = np.digitize(nonzero_values, bins, right=True)
        random_values = self._np_rng.random(len(nonzero_values))
        digits = np.ceil(random_values * (right - left) + left).astype(np.int64)
        digits = np.clip(digits, 1, self.num_bins - 1)
        output = np.zeros_like(values_np, dtype=np.float32)
        output[nonzero_mask] = digits.astype(np.float32)
        return torch.from_numpy(output)

    def _encode_values(self, values):
        if self.contextual_value_mode == "bin":
            return self._bin_values(values)
        if self.contextual_value_mode in {"as_is", "raw", "none"}:
            return values.detach().float().cpu()
        if self.contextual_value_mode == "log1p":
            return torch.log1p(torch.clamp(values.detach().float().cpu(), min=0.0))
        raise ValueError(
            f"Unknown contextual_value_mode: {self.contextual_value_mode}"
        )

    def _select_positions(self, row, max_gene_tokens):
        matched_device = self._matched_gene_indices.to(row.device)
        matched_values = row.index_select(0, matched_device)
        candidate_local = torch.nonzero(matched_values != 0, as_tuple=False).flatten()
        if candidate_local.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        if candidate_local.numel() > max_gene_tokens:
            if self.contextual_gene_selection == "sample":
                permutation = torch.randperm(
                    candidate_local.numel(),
                    generator=self._torch_generator,
                )[:max_gene_tokens]
                candidate_local = candidate_local.detach().cpu().index_select(
                    0, permutation
                )
            elif self.contextual_gene_selection == "top_expression":
                candidate_values = matched_values.index_select(
                    0, candidate_local.to(matched_values.device)
                ).abs()
                top = torch.topk(
                    candidate_values,
                    k=max_gene_tokens,
                    largest=True,
                    sorted=False,
                ).indices
                candidate_local = candidate_local.index_select(0, top).detach().cpu()
            else:
                candidate_local = candidate_local[:max_gene_tokens].detach().cpu()
        else:
            candidate_local = candidate_local.detach().cpu()

        return self._matched_gene_indices.index_select(0, candidate_local)

    def _build_contextual_fallback(self, batch_size):
        if self.contextual_fallback == "zero":
            base = torch.zeros(
                len(self.gene_token_ids),
                self.gears_hidden_size,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            if self._static_gene_embeddings_device_cache is None:
                self._static_gene_embeddings_device_cache = (
                    self.get_static_gene_embeddings(self.gene_list)
                    .to(self.device, dtype=torch.float32)
                    .detach()
                )
            base = self._static_gene_embeddings_device_cache
        return base.unsqueeze(0).expand(batch_size, -1, -1).contiguous().clone()

    def _autocast_settings(self):
        enabled = self.device.startswith("cuda") and self.precision != "fp32"
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        return enabled, dtype, device_type

    def get_contextual_gene_embeddings(self, x: torch.Tensor, gene_list: List[str]):
        if self.gene_token_ids is None:
            raise RuntimeError("Call setup() before requesting contextual embeddings.")
        if x.ndim != 2 or x.shape[1] != len(self.gene_token_ids):
            raise ValueError(
                "Expected x with shape [batch, num_genes] = "
                f"[batch, {len(self.gene_token_ids)}], received {tuple(x.shape)}"
            )
        if torch.any(x < 0):
            raise ValueError("scGPT contextual input must be non-negative.")

        self._ensure_encoder_loaded()
        batch_size = int(x.shape[0])
        full_embeddings = self._build_contextual_fallback(batch_size)
        max_gene_tokens = self.max_seq_len - (1 if self.use_cls else 0)
        if max_gene_tokens <= 0:
            raise ValueError("max_seq_len leaves no room for gene tokens.")

        sequences = []
        active_batch_indices = []
        selected_positions = []
        for batch_index in range(batch_size):
            positions = self._select_positions(x[batch_index], max_gene_tokens)
            if positions.numel() == 0:
                continue
            values = x[batch_index].index_select(0, positions.to(x.device))
            token_ids = self._gene_token_tensor.index_select(0, positions)
            encoded_values = self._encode_values(values)

            if self.use_cls:
                token_ids = torch.cat(
                    [torch.tensor([int(self.cls_token_id)]), token_ids], dim=0
                )
                encoded_values = torch.cat(
                    [torch.tensor([self.pad_value]), encoded_values], dim=0
                )

            sequences.append((token_ids, encoded_values))
            active_batch_indices.append(batch_index)
            selected_positions.append(positions)

        if not sequences:
            return full_embeddings

        sequence_length = max(len(token_ids) for token_ids, _ in sequences)
        active_count = len(sequences)
        input_ids = torch.full(
            (active_count, sequence_length),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        input_values = torch.full(
            (active_count, sequence_length),
            self.pad_value,
            dtype=torch.float32,
            device=self.device,
        )
        key_padding_mask = torch.ones(
            (active_count, sequence_length),
            dtype=torch.bool,
            device=self.device,
        )
        for row_index, (token_ids, values) in enumerate(sequences):
            length = len(token_ids)
            input_ids[row_index, :length] = token_ids.to(self.device)
            input_values[row_index, :length] = values.to(self.device)
            key_padding_mask[row_index, :length] = False

        encoder_batch_size = self.contextual_encoder_batch_size or active_count
        output_chunks = []
        autocast_enabled, autocast_dtype, autocast_device_type = (
            self._autocast_settings()
        )
        with torch.no_grad():
            for start in range(0, active_count, encoder_batch_size):
                end = min(start + encoder_batch_size, active_count)
                with torch.autocast(
                    device_type=autocast_device_type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    chunk = self.scgpt_encoder(
                        input_ids[start:end],
                        input_values[start:end],
                        key_padding_mask[start:end],
                    )
                output_chunks.append(chunk.detach().float())
        contextual = torch.cat(output_chunks, dim=0)
        contextual = self._project_to_hidden(contextual)

        offset = 1 if self.use_cls else 0
        for row_index, (batch_index, positions) in enumerate(
            zip(active_batch_indices, selected_positions)
        ):
            length = len(positions)
            full_embeddings[
                batch_index, positions.to(self.device), :
            ] = contextual[row_index, offset : offset + length, :]

        return full_embeddings

    def get_config(self):
        return {
            "adapter_name": self.name,
            "model_dir": str(self.model_dir) if self.model_dir is not None else None,
            "ckpt_path": str(self.ckpt_path),
            "vocab_path": str(self.vocab_path),
            "args_path": str(self.args_path),
            "scgpt_source_dir": (
                str(self.scgpt_source_dir)
                if self.scgpt_source_dir is not None
                else None
            ),
            "device": self.device,
            "output_dim": self.output_dim,
            "raw_output_dim": self.raw_output_dim,
            "max_seq_len": self.max_seq_len,
            "num_bins": self.num_bins,
            "contextual_value_mode": self.contextual_value_mode,
            "contextual_gene_selection": self.contextual_gene_selection,
            "contextual_fallback": self.contextual_fallback,
            "contextual_encoder_batch_size": self.contextual_encoder_batch_size,
            "missing_strategy": self.missing_strategy,
            "project_method": self.project_method,
            "normalize": self.normalize,
            "precision": self.precision,
            "seed": self.seed,
            "flag_embedding_shape": self.flag_embedding_shape,
        }
