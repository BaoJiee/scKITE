import json
from pathlib import Path
from typing import List, Optional
import importlib.util
import sys

import torch

from .base import BaseSCFMAdapter

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
    bucket_ids = torch.bucketize(pos_values, inner_edges, right=False) + 1
    out[pos_mask] = bucket_ids.long()
    return out



class ScKITEAdapter(BaseSCFMAdapter):
    name = "sckite"

    def __init__(
        self,
        ckpt_path,
        vocab_path,
        device="cuda",
        gears_hidden_size=64,
        missing_strategy="mean_gene",
        project_method="slice",
        normalize=False,
        model_py_path=None,
        model_kwargs=None,
        num_bins=51,
        contextual_value_mode="bin",
        contextual_max_genes=None,
        contextual_gene_selection="matched_first",
        contextual_fallback="static",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.ckpt_path = Path(ckpt_path)
        self.vocab_path = Path(vocab_path)
        self.gears_hidden_size = int(gears_hidden_size)
        self.output_dim = int(gears_hidden_size)
        self.raw_output_dim = None
        self.missing_strategy = str(missing_strategy)
        self.project_method = str(project_method)
        self.normalize = bool(normalize)

        self.model_py_path = Path(model_py_path) if model_py_path is not None else None
        self.model_kwargs = model_kwargs or {}
        self.num_bins = int(num_bins)
        self.contextual_value_mode = str(contextual_value_mode)
        self.contextual_max_genes = None if contextual_max_genes is None else int(contextual_max_genes)
        self.contextual_gene_selection = str(contextual_gene_selection)
        self.contextual_fallback = str(contextual_fallback)
        self.sckite_model = None

        self.symbol_to_id = {}
        self.symbol_upper_to_id = {}
        self.ensembl_to_id = {}
        self.token_to_id = {}
        self.gene_token_ids_all = []
        self.shared_embedding = None
        self.gene_list = None
        self.pert_list = None
        self.gene_token_ids = None
        self.pert_token_ids = None

        self._load_vocab()
        self._load_shared_embedding()

    def _read_json_or_jsonl(self, path):
        text = Path(path).read_text(encoding="utf-8")
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return obj.get("genes", obj.get("items", obj.get("data", [])))
            return []
        except Exception:
            return [json.loads(line) for line in text.splitlines() if line.strip()]

    def _load_vocab(self):
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"vocab_path not found: {self.vocab_path}")
        records = self._read_json_or_jsonl(self.vocab_path)
        if not isinstance(records, list):
            raise ValueError("vocab_path must be a jsonl file or a json file containing a list.")
        for item in records:
            if not isinstance(item, dict):
                continue
            token_id = item.get("token_id", item.get("id", item.get("global_id", None)))
            gene_symbol = item.get("gene_symbol", item.get("symbol", None))
            ensembl_id = item.get("ensembl_id", item.get("ensembl", None))
            token = item.get("token", item.get("global_token", None))
            if token_id is None:
                continue
            token_id = int(token_id)
            self.gene_token_ids_all.append(token_id)
            if token is not None:
                self.token_to_id[str(token)] = token_id
            if gene_symbol is not None:
                gene_symbol = str(gene_symbol)
                self.symbol_to_id[gene_symbol] = token_id
                self.symbol_upper_to_id[gene_symbol.upper()] = token_id
                self.token_to_id[f"<gene:{gene_symbol}>"] = token_id
                self.token_to_id[f"<gene:{gene_symbol.upper()}>"] = token_id
            if ensembl_id is not None:
                ensembl_id = str(ensembl_id)
                self.ensembl_to_id[ensembl_id] = token_id
                self.token_to_id[f"<gene:{ensembl_id}>"] = token_id
        self.gene_token_ids_all = sorted(set(self.gene_token_ids_all))
        print(f"[ScKITEAdapter] vocab records={len(records)}, gene_tokens={len(self.gene_token_ids_all)}, symbols={len(self.symbol_to_id)}, ensembl={len(self.ensembl_to_id)}")

    def _extract_state_dict(self, ckpt):
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        if isinstance(ckpt, dict) and "model" in ckpt:
            return ckpt["model"]
        return ckpt

    def _load_shared_embedding(self):
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"ckpt_path not found: {self.ckpt_path}")
        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        state = self._extract_state_dict(ckpt)
        state = {str(k).replace("module.", ""): v for k, v in state.items()}
        if "shared_token_embedding.weight" not in state:
            keys = [k for k in state.keys() if "embedding" in str(k).lower()]
            raise KeyError(f"Cannot find shared_token_embedding.weight. Embedding-like keys: {keys[:20]}")
        emb = state["shared_token_embedding.weight"].detach().float().cpu()
        self.shared_embedding = emb
        self.raw_output_dim = int(emb.shape[1])
        print(f"[ScKITEAdapter] shared_token_embedding shape={tuple(emb.shape)}, adapter output_dim={self.output_dim}")

    def setup(self, gene_list: List[str], pert_list: List[str], gene_id_list=None):
        self.gene_list = list(gene_list)
        self.pert_list = list(pert_list)
        self.gene_id_list = list(gene_id_list) if gene_id_list is not None else None

        self.gene_token_ids = self._map_gene_tokens_by_best_coverage(
            gene_list=self.gene_list,
            gene_id_list=self.gene_id_list,
        )

        self.pert_token_ids = self._map_names_to_token_ids(self.pert_list)

        gene_matched = sum(x is not None for x in self.gene_token_ids)
        pert_matched = sum(x is not None for x in self.pert_token_ids)

        print(f"[ScKITEAdapter] GEARS genes matched={gene_matched}/{len(self.gene_token_ids)}, missing={len(self.gene_token_ids) - gene_matched}")
        print(f"[ScKITEAdapter] GEARS perts matched={pert_matched}/{len(self.pert_token_ids)}, missing={len(self.pert_token_ids) - pert_matched}")

        if gene_matched == 0:
            raise ValueError("No GEARS gene matched the scKITE vocabulary. Please check gene_symbol / ensembl_id format.")

        if pert_matched == 0:
            raise ValueError("No GEARS perturbation gene matched the scKITE vocabulary. Please check pert_list format.")

    def _lookup_token_id(self, name: str) -> Optional[int]:
        if name is None:
            return None
        name = str(name)
        if name in self.symbol_to_id:
            return self.symbol_to_id[name]
        if name.upper() in self.symbol_upper_to_id:
            return self.symbol_upper_to_id[name.upper()]
        if name in self.ensembl_to_id:
            return self.ensembl_to_id[name]
        if name in self.token_to_id:
            return self.token_to_id[name]
        if f"<gene:{name}>" in self.token_to_id:
            return self.token_to_id[f"<gene:{name}>"]
        if f"<gene:{name.upper()}>" in self.token_to_id:
            return self.token_to_id[f"<gene:{name.upper()}>"]
        return None

    def _map_names_to_token_ids(self, names: List[str]) -> List[Optional[int]]:
        return [self._lookup_token_id(x) for x in names]


    def _strip_version(self, name):
        if name is None:
            return None
        name = str(name)
        if "." in name:
            left, right = name.rsplit(".", 1)
            if right.isdigit():
                return left
        return name

    def _lookup_gene_token_by_strategy(self, gene_name, gene_id, strategy):
        gene_name = None if gene_name is None else str(gene_name)
        gene_id = None if gene_id is None else str(gene_id)

        if strategy == "var_index_ensembl":
            if gene_id in self.ensembl_to_id:
                return self.ensembl_to_id[gene_id]
            if f"<gene:{gene_id}>" in self.token_to_id:
                return self.token_to_id[f"<gene:{gene_id}>"]
            return None

        if strategy == "var_index_ensembl_strip":
            gene_id_strip = self._strip_version(gene_id)
            if gene_id_strip in self.ensembl_to_id:
                return self.ensembl_to_id[gene_id_strip]
            if f"<gene:{gene_id_strip}>" in self.token_to_id:
                return self.token_to_id[f"<gene:{gene_id_strip}>"]
            return None

        if strategy == "gene_symbol_exact":
            if gene_name in self.symbol_to_id:
                return self.symbol_to_id[gene_name]
            if f"<gene:{gene_name}>" in self.token_to_id:
                return self.token_to_id[f"<gene:{gene_name}>"]
            return None

        if strategy == "gene_symbol_upper":
            gene_name_upper = None if gene_name is None else gene_name.upper()
            if gene_name_upper in self.symbol_upper_to_id:
                return self.symbol_upper_to_id[gene_name_upper]
            if f"<gene:{gene_name_upper}>" in self.token_to_id:
                return self.token_to_id[f"<gene:{gene_name_upper}>"]
            return None

        raise ValueError(f"Unknown gene match strategy: {strategy}")

    def _map_gene_tokens_by_best_coverage(self, gene_list, gene_id_list=None):
        gene_list = list(gene_list)
        gene_id_list = list(gene_id_list) if gene_id_list is not None else [None] * len(gene_list)

        if len(gene_list) != len(gene_id_list):
            raise ValueError(f"gene_list and gene_id_list length mismatch: {len(gene_list)} vs {len(gene_id_list)}")

        strategies = [
            "var_index_ensembl",
            "var_index_ensembl_strip",
            "gene_symbol_exact",
            "gene_symbol_upper",
        ]

        if all(x is None for x in gene_id_list):
            strategies = ["gene_symbol_exact", "gene_symbol_upper"]

        strategy_to_token_ids = {}
        strategy_summary = []

        for strategy in strategies:
            token_ids = [
                self._lookup_gene_token_by_strategy(gene_name, gene_id, strategy)
                for gene_name, gene_id in zip(gene_list, gene_id_list)
            ]
            matched = sum(x is not None for x in token_ids)
            strategy_to_token_ids[strategy] = token_ids
            strategy_summary.append({"strategy": strategy, "matched": matched, "total": len(gene_list), "rate": matched / len(gene_list)})

        strategy_summary = sorted(strategy_summary, key=lambda x: x["matched"], reverse=True)
        ordered_strategies = [x["strategy"] for x in strategy_summary]

        final_token_ids = [None] * len(gene_list)
        final_sources = ["missing"] * len(gene_list)

        for strategy in ordered_strategies:
            token_ids = strategy_to_token_ids[strategy]
            for i, token_id in enumerate(token_ids):
                if final_token_ids[i] is None and token_id is not None:
                    final_token_ids[i] = token_id
                    final_sources[i] = strategy

        final_matched = sum(x is not None for x in final_token_ids)
        self.gene_match_summary = strategy_summary
        self.gene_match_sources = final_sources

        print("[ScKITEAdapter] gene matching strategy coverage:")
        for item in strategy_summary:
            print(f"  {item['strategy']}: {item['matched']}/{item['total']} = {item['rate']:.4f}")
        print(f"[ScKITEAdapter] selected primary gene strategy: {ordered_strategies[0]}")
        print(f"[ScKITEAdapter] final gene matched after supplement: {final_matched}/{len(gene_list)} = {final_matched / len(gene_list):.4f}")

        return final_token_ids


    def _missing_fill_vector(self) -> torch.Tensor:
        if self.missing_strategy == "zero":
            return torch.zeros(self.raw_output_dim)
        if self.missing_strategy == "mean_all":
            return self.shared_embedding.mean(dim=0)
        valid_ids = [i for i in self.gene_token_ids_all if 0 <= int(i) < self.shared_embedding.shape[0]]
        if len(valid_ids) == 0:
            return self.shared_embedding.mean(dim=0)
        return self.shared_embedding[torch.tensor(valid_ids, dtype=torch.long)].mean(dim=0)

    def _embedding_by_token_ids(self, token_ids: List[Optional[int]]) -> torch.Tensor:
        fill = self._missing_fill_vector()
        max_id = self.shared_embedding.shape[0] - 1
        out = []
        for tid in token_ids:
            if tid is None or int(tid) < 0 or int(tid) > max_id:
                out.append(fill)
            else:
                out.append(self.shared_embedding[int(tid)])
        return torch.stack(out, dim=0)

    def _project_to_hidden(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.shape[1] == self.gears_hidden_size:
            out = emb
        elif self.project_method == "slice":
            if emb.shape[1] > self.gears_hidden_size:
                out = emb[:, : self.gears_hidden_size].contiguous()
            else:
                pad = torch.zeros(emb.shape[0], self.gears_hidden_size - emb.shape[1], dtype=emb.dtype, device=emb.device)
                out = torch.cat([emb, pad], dim=1)
        elif self.project_method == "mean_pool":
            if emb.shape[1] % self.gears_hidden_size != 0:
                raise ValueError("mean_pool requires raw_output_dim % gears_hidden_size == 0.")
            group = emb.shape[1] // self.gears_hidden_size
            out = emb.reshape(emb.shape[0], self.gears_hidden_size, group).mean(dim=2)
        else:
            raise ValueError(f"Unknown project_method: {self.project_method}")

        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2, dim=1)

        return out

    def get_static_gene_embeddings(self, gene_list: List[str]) -> torch.Tensor:
        if self.gene_token_ids is None:
            raise RuntimeError("Please call setup(gene_list, pert_list, gene_id_list) before get_static_gene_embeddings().")

        if len(gene_list) == len(self.gene_token_ids):
            token_ids = self.gene_token_ids
        else:
            token_ids = self._map_names_to_token_ids(list(gene_list))

        emb = self._embedding_by_token_ids(token_ids)
        emb = self._project_to_hidden(emb)
        return emb.clone()

    def get_pert_embeddings(self, pert_list: List[str]) -> torch.Tensor:
        if self.pert_list is None:
            raise RuntimeError("Please call setup(gene_list, pert_list, gene_id_list) before get_pert_embeddings().")

        token_ids = self._map_names_to_token_ids(list(pert_list))
        emb = self._embedding_by_token_ids(token_ids)
        emb = self._project_to_hidden(emb)
        return emb.clone()


    def _load_sckite_model_class(self):
        if self.model_py_path is None:
            raise ValueError("sckite_contextual requires model_py_path.")
        if not self.model_py_path.exists():
            raise FileNotFoundError(f"model_py_path not found: {self.model_py_path}")

        model_dir = str(self.model_py_path.parent)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        spec = importlib.util.spec_from_file_location("sckite_model_module", str(self.model_py_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class_name = getattr(self, "model_class_name", "ScKITEStage2Model")
        if not hasattr(module, class_name):
            available = [x for x in dir(module) if "Model" in x or "model" in x]
            raise AttributeError(f"Cannot find class {class_name} in {self.model_py_path}. Available model-like names: {available}")

        return getattr(module, class_name)

    def _ensure_sckite_model_loaded(self):
        if self.sckite_model is not None:
            return

        model_cls = self._load_sckite_model_class()
        model_kwargs = dict(getattr(self, "model_kwargs", {}) or {})

        model_kwargs.setdefault("global_vocab_size", int(self.shared_embedding.shape[0]))
        model_kwargs.setdefault("d_model", int(self.raw_output_dim))

        print(f"[ScKITEAdapter] loading contextual scKITE model with kwargs={model_kwargs}")
        model = model_cls(**model_kwargs)

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        state = self._extract_state_dict(ckpt)
        state = {str(k).replace("module.", ""): v for k, v in state.items()}

        model_state = model.state_dict()
        compatible_state = {}
        skipped_shape = []
        ckpt_unexpected_keys = []

        for k, v in state.items():
            if k not in model_state:
                ckpt_unexpected_keys.append(k)
                continue
            if hasattr(v, "shape") and tuple(v.shape) != tuple(model_state[k].shape):
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
            compatible_state[k] = v






        critical_encoder_prefixes = (
            "shared_token_embedding.",
            "token_norm.",
            "value_encoder.",
            "mask_flag_embedding.",
            "encoder.",
        )

        def is_critical_encoder_key(key: str) -> bool:
            return str(key).startswith(critical_encoder_prefixes)


        bad_encoder_shape = [
            item for item in skipped_shape
            if is_critical_encoder_key(item[0])
        ]

        if bad_encoder_shape:
            details = "\n".join(
                f"  {k}: ckpt={ckpt_shape}, model={model_shape}"
                for k, ckpt_shape, model_shape in bad_encoder_shape[:100]
            )
            raise RuntimeError(
                "[ScKITEAdapter] Critical Encoder parameters have shape mismatch. "
                "Contextual model loading is aborted.\n"
                f"{details}"
            )



        expected_encoder_keys = [
            k for k in model_state.keys()
            if is_critical_encoder_key(k)
        ]

        not_loaded_encoder_keys = [
            k for k in expected_encoder_keys
            if k not in compatible_state
        ]

        if not_loaded_encoder_keys:
            details = "\n".join(f"  {k}" for k in not_loaded_encoder_keys[:200])
            raise RuntimeError(
                "[ScKITEAdapter] Some critical Encoder parameters are missing "
                "from the checkpoint or were not compatible. "
                "Contextual model loading is aborted.\n"
                f"{details}"
            )


        missing, unexpected = model.load_state_dict(compatible_state, strict=False)

        print("[ScKITEAdapter] contextual model loaded with safe filtering.")
        print(
            f"[ScKITEAdapter] loaded keys={len(compatible_state)}, "
            f"missing={len(missing)}, "
            f"unexpected={len(unexpected)}, "
            f"skipped_shape={len(skipped_shape)}, "
            f"ckpt_unexpected={len(ckpt_unexpected_keys)}"
        )

        if len(skipped_shape) > 0:
            print("[ScKITEAdapter] first shape-mismatched skipped keys:")
            for item in skipped_shape[:20]:
                print(f"  {item[0]}: ckpt={item[1]}, model={item[2]}")


        missing_encoder_keys = [
            k for k in missing
            if is_critical_encoder_key(k)
        ]

        if missing_encoder_keys:
            details = "\n".join(f"  {k}" for k in missing_encoder_keys[:200])
            raise RuntimeError(
                "[ScKITEAdapter] Critical Encoder parameters were reported as missing "
                "after load_state_dict. Contextual model loading is aborted.\n"
                f"{details}"
            )



        loaded_model_state = model.state_dict()
        verification_failed = []

        for key in expected_encoder_keys:
            ckpt_tensor = compatible_state[key]
            model_tensor = loaded_model_state[key]

            if not torch.is_tensor(ckpt_tensor) or not torch.is_tensor(model_tensor):
                verification_failed.append((key, "non-tensor state value"))
                continue

            ckpt_tensor_cpu = ckpt_tensor.detach().cpu()
            model_tensor_cpu = model_tensor.detach().cpu()

            if (
                ckpt_tensor_cpu.shape != model_tensor_cpu.shape
                or ckpt_tensor_cpu.dtype != model_tensor_cpu.dtype
                or not torch.equal(ckpt_tensor_cpu, model_tensor_cpu)
            ):
                verification_failed.append(
                    (
                        key,
                        f"ckpt_shape={tuple(ckpt_tensor_cpu.shape)}, "
                        f"model_shape={tuple(model_tensor_cpu.shape)}, "
                        f"ckpt_dtype={ckpt_tensor_cpu.dtype}, "
                        f"model_dtype={model_tensor_cpu.dtype}",
                    )
                )

        if verification_failed:
            details = "\n".join(
                f"  {key}: {reason}"
                for key, reason in verification_failed[:100]
            )
            raise RuntimeError(
                "[ScKITEAdapter] Exact tensor verification failed for critical "
                "Encoder parameters. Contextual model loading is aborted.\n"
                f"{details}"
            )

        loaded_encoder_keys = [
            k for k in compatible_state.keys()
            if is_critical_encoder_key(k)
        ]

        print(
            "[ScKITEAdapter] Encoder loading check passed: "
            f"{len(loaded_encoder_keys)}/{len(expected_encoder_keys)} "
            "critical Encoder parameters loaded."
        )
        print(
            "[ScKITEAdapter] Exact tensor verification passed for all "
            "critical Encoder parameters."
        )


        decoder_or_other_missing = [
            k for k in missing
            if not is_critical_encoder_key(k)
        ]
        decoder_or_other_shape_skips = [
            item for item in skipped_shape
            if not is_critical_encoder_key(item[0])
        ]

        if decoder_or_other_missing:
            print(
                "[ScKITEAdapter] non-Encoder missing keys are allowed "
                f"for decoder algorithm changes: {len(decoder_or_other_missing)}"
            )
            for k in decoder_or_other_missing[:20]:
                print(f"  missing non-Encoder key: {k}")

        if decoder_or_other_shape_skips:
            print(
                "[ScKITEAdapter] non-Encoder shape mismatches are allowed "
                f"for decoder algorithm changes: {len(decoder_or_other_shape_skips)}"
            )

        if ckpt_unexpected_keys:
            print(
                "[ScKITEAdapter] checkpoint-only keys are allowed "
                f"for decoder algorithm changes: {len(ckpt_unexpected_keys)}"
            )
            for k in ckpt_unexpected_keys[:20]:
                print(f"  checkpoint-only key: {k}")

        model.to(self.device)

        if str(self.device).startswith("cuda"):
            model.half()

        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        self.sckite_model = model
    def _select_contextual_gene_indices(self, token_ids: List[Optional[int]], n_genes: int) -> torch.Tensor:
        if self.contextual_max_genes is None:
            return torch.arange(n_genes, dtype=torch.long)
        if self.contextual_max_genes <= 0:
            return torch.arange(n_genes, dtype=torch.long)
        if self.contextual_max_genes >= n_genes:
            return torch.arange(n_genes, dtype=torch.long)

        mode = self.contextual_gene_selection.lower()
        max_genes = int(self.contextual_max_genes)

        if mode == "first":
            indices = list(range(max_genes))
            return torch.tensor(indices, dtype=torch.long)

        if mode == "matched_first":
            matched = [i for i, tid in enumerate(token_ids) if tid is not None]
            missing = [i for i, tid in enumerate(token_ids) if tid is None]
            indices = (matched + missing)[:max_genes]
            return torch.tensor(indices, dtype=torch.long)

        raise ValueError(f"Unknown contextual_gene_selection: {self.contextual_gene_selection}")

    def _build_contextual_inputs(self, x: torch.Tensor, gene_list: List[str]):
        device = self.device
        x = x.to(device).float()
        n_genes = len(gene_list)

        if self.gene_token_ids is not None and len(gene_list) == len(self.gene_token_ids):
            token_ids = self.gene_token_ids
        else:
            token_ids = self._map_names_to_token_ids(list(gene_list))

        selected_idx_cpu = self._select_contextual_gene_indices(token_ids, n_genes)
        selected_idx = selected_idx_cpu.to(device)

        x_selected = x.index_select(dim=1, index=selected_idx)
        token_ids_selected = [token_ids[i] for i in selected_idx_cpu.tolist()]

        pad_token_id = int(getattr(self.sckite_model, "pad_token_id", 0))
        input_ids = [int(tid) if tid is not None else pad_token_id for tid in token_ids_selected]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        input_ids = input_ids.unsqueeze(0).expand(x.shape[0], -1).contiguous()

        missing_mask = torch.tensor([tid is None for tid in token_ids_selected], dtype=torch.bool, device=device)
        key_padding_mask = missing_mask.unsqueeze(0).expand(x.shape[0], -1).contiguous()

        encoder_values = self._encode_contextual_values(x_selected)

        return input_ids, encoder_values, key_padding_mask, selected_idx, token_ids

    def _build_contextual_fallback_embeddings(self, token_ids: List[Optional[int]], batch_size: int) -> torch.Tensor:
        device = self.device
        n_genes = len(token_ids)

        if self.contextual_fallback.lower() == "zero":
            full_emb = torch.zeros(n_genes, self.gears_hidden_size, dtype=torch.float32, device=device)
        elif self.contextual_fallback.lower() == "static":
            raw_emb = self._embedding_by_token_ids(token_ids).to(device).float()
            full_emb = self._project_to_hidden(raw_emb).to(device).float()
        else:
            raise ValueError(f"Unknown contextual_fallback: {self.contextual_fallback}")

        full_emb = full_emb.unsqueeze(0).expand(batch_size, -1, -1).contiguous().clone()
        return full_emb

    def _encode_contextual_values(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.contextual_value_mode.lower()
        if mode == "bin":
            return left_binning(x, self.num_bins).float()
        if mode in ["as_is", "none", "raw"]:
            return x.float()
        if mode == "log1p":
            return torch.log1p(torch.clamp(x.float(), min=0.0))
        raise ValueError(f"Unknown contextual_value_mode: {self.contextual_value_mode}")

    def get_contextual_gene_embeddings(self, x: torch.Tensor, gene_list: List[str]):
        self._ensure_sckite_model_loaded()
        input_ids, encoder_values, key_padding_mask, selected_idx, token_ids = self._build_contextual_inputs(x, gene_list)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
                encoder_outputs, _ = self.sckite_model.encode(
                    encoder_input_gene_ids=input_ids,
                    encoder_input_values=encoder_values,
                    encoder_key_padding_mask=key_padding_mask,
                    return_last_attn=False,
                )

        encoder_outputs = encoder_outputs.detach().float().clone()
        batch_size = encoder_outputs.shape[0]
        selected_gene_num = encoder_outputs.shape[1]
        raw_dim = encoder_outputs.shape[2]

        contextual_emb = encoder_outputs.reshape(batch_size * selected_gene_num, raw_dim)
        contextual_emb = self._project_to_hidden(contextual_emb)
        contextual_emb = contextual_emb.reshape(batch_size, selected_gene_num, self.gears_hidden_size)

        if selected_gene_num == len(gene_list):
            return contextual_emb

        full_emb = self._build_contextual_fallback_embeddings(token_ids, batch_size=batch_size)
        full_emb[:, selected_idx, :] = contextual_emb

        return full_emb
