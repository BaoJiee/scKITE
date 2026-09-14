import copy
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from .base import BaseSCFMAdapter


class ScFoundationAdapter(BaseSCFMAdapter):
    """Frozen scFoundation gene-context embeddings for the existing GEARS flow.

    The adapter keeps GEARS' PertData, splits, cell graphs, model, loss and
    evaluation unchanged.  It aligns GEARS genes to scFoundation's fixed 19,264
    genes, calls the official ``MaeAutobin`` implementation, and gathers the
    result back to GEARS' original gene order.
    """

    name = "scfoundation"

    DEFAULT_HGNC_SHA256 = (
        "2106d1f237d6c542a85a4c399225e011ee9c5822199e7a400b7b9f842c8c8ca0"
    )

    VERIFIED_ALIASES = {
        "C19ORF26": "CBARP",
        "C3ORF72": "FOXL2NB",
        "ELMSAN1": "MIDEAS",
        "KIAA1804": "MAP3K21",
        "RHOXF2BB": "RHOXF2B",
    }

    _METHOD_PRIORITY = {
        "exact_symbol": 0,
        "case_insensitive_symbol": 1,
        "verified_alias": 2,
        "ensembl_hgnc": 3,
        "hgnc_approved": 4,
        "hgnc_previous": 5,
        "hgnc_alias": 6,
    }

    def __init__(
        self,
        source_dir,
        ckpt_path,
        canonical_gene_path,
        hgnc_mapping_path,
        device="cuda",
        gears_hidden_size=512,
        checkpoint_key="gene",
        pre_normalized="T",
        target_high_resolution=4.0,
        contextual_fallback="static",
        contextual_encoder_batch_size=1,
        missing_strategy="mean_gene",
        precision="fp16",
        mapping_report_dir=None,
        expected_hgnc_sha256=DEFAULT_HGNC_SHA256,
        verify_loaded_tensors=True,
        require_all_perturbations=True,
        required_perturbation_genes=None,
        seed=1,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.source_dir = Path(source_dir)
        self.model_code_dir = self.source_dir / "model"
        self.ckpt_path = Path(ckpt_path)
        self.canonical_gene_path = Path(canonical_gene_path)
        self.hgnc_mapping_path = Path(hgnc_mapping_path)
        self.mapping_report_dir = (
            Path(mapping_report_dir) if mapping_report_dir is not None else None
        )
        self.expected_hgnc_sha256 = (
            str(expected_hgnc_sha256).strip().lower()
            if expected_hgnc_sha256
            else None
        )

        required_paths = (
            ("scFoundation source", self.source_dir),
            ("scFoundation model code", self.model_code_dir),
            ("scFoundation checkpoint", self.ckpt_path),
            ("scFoundation canonical gene list", self.canonical_gene_path),
            ("HGNC mapping", self.hgnc_mapping_path),
            ("scFoundation loader", self.model_code_dir / "load.py"),
        )
        for label, path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        self.device = str(device)
        self.gears_hidden_size = int(gears_hidden_size)
        self.output_dim = self.gears_hidden_size
        self.checkpoint_key = str(checkpoint_key)
        self.pre_normalized = str(pre_normalized).upper()
        self.target_high_resolution = float(target_high_resolution)
        self.contextual_fallback = str(contextual_fallback).lower()
        self.contextual_encoder_batch_size = max(
            1, int(contextual_encoder_batch_size)
        )
        self.missing_strategy = str(missing_strategy).lower()
        self.precision = str(precision).lower()
        self.verify_loaded_tensors = bool(verify_loaded_tensors)
        self.require_all_perturbations = bool(require_all_perturbations)
        self.required_perturbation_genes = (
            sorted({str(gene) for gene in required_perturbation_genes})
            if required_perturbation_genes is not None
            else None
        )
        self.seed = int(seed)

        if self.checkpoint_key != "gene":
            raise ValueError(
                "GEARS gene-context embeddings require checkpoint_key='gene'."
            )
        if self.pre_normalized != "T":
            raise ValueError(
                "The unchanged GEARS cell graph contains normalized+log1p x but no "
                "raw total count; this adapter therefore requires pre_normalized='T'."
            )
        if self.contextual_fallback not in {"static", "zero"}:
            raise ValueError("contextual_fallback must be 'static' or 'zero'.")
        if self.missing_strategy not in {"mean_gene", "zero"}:
            raise ValueError("missing_strategy must be 'mean_gene' or 'zero'.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16")

        self.hgnc_sha256 = self._sha256(self.hgnc_mapping_path)
        if (
            self.expected_hgnc_sha256 is not None
            and self.hgnc_sha256 != self.expected_hgnc_sha256
        ):
            raise ValueError(
                "HGNC mapping checksum mismatch: "
                f"expected={self.expected_hgnc_sha256}, actual={self.hgnc_sha256}, "
                f"path={self.hgnc_mapping_path}"
            )

        self.canonical_genes = self._load_canonical_genes(
            self.canonical_gene_path
        )
        self.canonical_gene_count = len(self.canonical_genes)
        if self.canonical_gene_count != 19264:
            raise ValueError(
                "scFoundation requires exactly 19,264 canonical genes; received "
                f"{self.canonical_gene_count} from {self.canonical_gene_path}."
            )

        self._canonical_exact: Dict[str, int] = {}
        self._canonical_upper: Dict[str, List[int]] = defaultdict(list)
        for index, symbol in enumerate(self.canonical_genes):
            if symbol in self._canonical_exact:
                raise ValueError(f"Duplicate canonical gene symbol: {symbol}")
            self._canonical_exact[symbol] = index
            self._canonical_upper[symbol.upper()].append(index)

        self._load_hgnc_mapping()

        self.scfoundation_model = None
        self.model_config = None
        self.loaded_tensor_count = 0
        self.encoder_hidden_dim = None
        self.decoder_hidden_dim = None
        self.seq_len = None
        self.pad_token_id = None
        self.mask_token_id = None
        self.bin_num = None

        self.gene_list = None
        self.gene_id_list = None
        self.pert_list = None
        self.gene_token_ids = None
        self.pert_token_ids = None
        self.gene_mapping_rows = None
        self.pert_mapping_rows = None
        self.required_pert_mapping_rows = None
        self.mapping_summary = None

        self._matched_gene_indices = None
        self._matched_token_indices = None
        self._static_canonical_embeddings_cpu = None
        self._static_aligned_embeddings_cpu = None
        self._static_aligned_embeddings_device = None
        self._missing_fill_cpu = None

        print(
            "[ScFoundationAdapter] "
            f"canonical_genes={self.canonical_gene_count}, "
            f"HGNC_SHA256={self.hgnc_sha256}, "
            f"checkpoint_key={self.checkpoint_key}, "
            f"pre_normalized={self.pre_normalized}, "
            f"target_high_resolution={self.target_high_resolution}"
        )

    def __deepcopy__(self, memo):
        """Share frozen model and immutable embedding caches in GEARS snapshots."""
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        shared = {
            "scfoundation_model",
            "_matched_gene_indices",
            "_matched_token_indices",
            "_static_canonical_embeddings_cpu",
            "_static_aligned_embeddings_cpu",
            "_static_aligned_embeddings_device",
            "_missing_fill_cpu",
        }
        for key, value in self.__dict__.items():
            if key in shared:
                setattr(result, key, value)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _strip_ensembl_version(value):
        if value is None:
            return None
        value = str(value).strip()
        left, separator, right = value.rpartition(".")
        if separator and right.isdigit():
            return left.upper()
        return value.upper()

    @staticmethod
    def _split_multi_value(value) -> List[str]:
        if value is None:
            return []
        value = str(value).strip()
        if not value or value.lower() == "nan":
            return []
        return [
            item.strip()
            for item in re.split(r"[|;,]", value)
            if item.strip()
        ]

    @staticmethod
    def _load_canonical_genes(path: Path) -> List[str]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "gene_name" not in reader.fieldnames:
                raise ValueError(
                    f"Canonical gene TSV must contain a gene_name column: {path}"
                )
            genes = [str(row["gene_name"]).strip() for row in reader]
        if not genes or any(not gene for gene in genes):
            raise ValueError(f"Canonical gene TSV contains empty gene names: {path}")
        return genes

    def _load_hgnc_mapping(self):
        self._hgnc_approved: Dict[str, Set[str]] = defaultdict(set)
        self._hgnc_previous: Dict[str, Set[str]] = defaultdict(set)
        self._hgnc_alias: Dict[str, Set[str]] = defaultdict(set)
        self._hgnc_ensembl: Dict[str, Set[str]] = defaultdict(set)
        self._hgnc_to_approved: Dict[str, str] = {}

        with self.hgnc_mapping_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or [])
            required = {"hgnc_id", "symbol", "ensembl_gene_id"}
            missing = sorted(required - fields)
            if missing:
                raise ValueError(
                    "HGNC mapping is missing required columns: " + ", ".join(missing)
                )

            record_count = 0
            for row in reader:
                hgnc_id = str(row.get("hgnc_id", "")).strip()
                approved = str(row.get("symbol", "")).strip()
                if not hgnc_id or not approved:
                    continue
                record_count += 1
                self._hgnc_to_approved[hgnc_id] = approved
                self._hgnc_approved[approved.upper()].add(hgnc_id)
                for symbol in self._split_multi_value(row.get("prev_symbol")):
                    self._hgnc_previous[symbol.upper()].add(hgnc_id)
                for symbol in self._split_multi_value(row.get("alias_symbol")):
                    self._hgnc_alias[symbol.upper()].add(hgnc_id)
                for ensembl_id in self._split_multi_value(
                    row.get("ensembl_gene_id")
                ):
                    normalized = self._strip_ensembl_version(ensembl_id)
                    if normalized:
                        self._hgnc_ensembl[normalized].add(hgnc_id)

        self._canonical_by_hgnc: Dict[str, Set[int]] = defaultdict(set)
        for canonical_index, symbol in enumerate(self.canonical_genes):
            upper = symbol.upper()
            hgnc_ids = self._hgnc_approved.get(upper, set())
            if not hgnc_ids:
                hgnc_ids = self._hgnc_previous.get(upper, set())
            if not hgnc_ids:
                hgnc_ids = self._hgnc_alias.get(upper, set())
            for hgnc_id in hgnc_ids:
                self._canonical_by_hgnc[hgnc_id].add(canonical_index)

        print(
            "[ScFoundationAdapter] HGNC records loaded: "
            f"records={record_count}, ensembl_ids={len(self._hgnc_ensembl)}, "
            f"canonical_hgnc_ids={len(self._canonical_by_hgnc)}"
        )

    def _canonical_candidates_from_hgnc_ids(
        self, hgnc_ids: Sequence[str]
    ) -> Set[int]:
        candidates: Set[int] = set()
        for hgnc_id in hgnc_ids:
            candidates.update(self._canonical_by_hgnc.get(hgnc_id, set()))
        return candidates

    def _single_candidate(self, candidates: Set[int]):
        return next(iter(candidates)) if len(candidates) == 1 else None

    def _resolved_symbol_for_hgnc_ids(self, hgnc_ids: Sequence[str]):
        approved = {
            self._hgnc_to_approved[hgnc_id]
            for hgnc_id in hgnc_ids
            if hgnc_id in self._hgnc_to_approved
        }
        return next(iter(approved)) if len(approved) == 1 else None

    def _resolve_gene(self, symbol, ensembl_id=None):
        symbol = "" if symbol is None else str(symbol).strip()
        upper = symbol.upper()
        ensembl_id = self._strip_ensembl_version(ensembl_id)

        if symbol in self._canonical_exact:
            index = self._canonical_exact[symbol]
            return index, "exact_symbol", self.canonical_genes[index], "matched"

        case_candidates = self._canonical_upper.get(upper, [])
        if len(case_candidates) == 1:
            index = case_candidates[0]
            return (
                index,
                "case_insensitive_symbol",
                self.canonical_genes[index],
                "matched",
            )

        alias_target = self.VERIFIED_ALIASES.get(upper)
        if alias_target is not None:
            alias_candidates = self._canonical_upper.get(alias_target.upper(), [])
            if len(alias_candidates) == 1:
                index = alias_candidates[0]
                return index, "verified_alias", self.canonical_genes[index], "matched"

        if ensembl_id:
            hgnc_ids = self._hgnc_ensembl.get(ensembl_id, set())
            candidates = self._canonical_candidates_from_hgnc_ids(hgnc_ids)
            index = self._single_candidate(candidates)
            if index is not None:
                return index, "ensembl_hgnc", self.canonical_genes[index], "matched"
            if len(candidates) > 1:
                return (
                    None,
                    "ensembl_hgnc",
                    self._resolved_symbol_for_hgnc_ids(hgnc_ids),
                    "ambiguous",
                )

        for method, mapping in (
            ("hgnc_approved", self._hgnc_approved),
            ("hgnc_previous", self._hgnc_previous),
            ("hgnc_alias", self._hgnc_alias),
        ):
            hgnc_ids = mapping.get(upper, set())
            candidates = self._canonical_candidates_from_hgnc_ids(hgnc_ids)
            index = self._single_candidate(candidates)
            if index is not None:
                return index, method, self.canonical_genes[index], "matched"
            if len(candidates) > 1:
                return (
                    None,
                    method,
                    self._resolved_symbol_for_hgnc_ids(hgnc_ids),
                    "ambiguous",
                )

        possible_hgnc = set()
        if ensembl_id:
            possible_hgnc.update(self._hgnc_ensembl.get(ensembl_id, set()))
        possible_hgnc.update(self._hgnc_approved.get(upper, set()))
        possible_hgnc.update(self._hgnc_previous.get(upper, set()))
        possible_hgnc.update(self._hgnc_alias.get(upper, set()))
        return (
            None,
            "none",
            self._resolved_symbol_for_hgnc_ids(possible_hgnc),
            "not_in_scfoundation",
        )

    def setup(
        self,
        gene_list: List[str],
        pert_list: List[str],
        gene_id_list: Optional[List[str]] = None,
    ):
        self.gene_list = list(gene_list)
        self.gene_id_list = (
            list(gene_id_list)
            if gene_id_list is not None
            else [None] * len(self.gene_list)
        )
        self.pert_list = list(pert_list)
        if len(self.gene_list) != len(self.gene_id_list):
            raise ValueError(
                "gene_list and gene_id_list length mismatch: "
                f"{len(self.gene_list)} vs {len(self.gene_id_list)}"
            )

        provisional = []
        for gene_index, (symbol, ensembl_id) in enumerate(
            zip(self.gene_list, self.gene_id_list)
        ):
            token_index, method, resolved_symbol, status = self._resolve_gene(
                symbol, ensembl_id
            )
            provisional.append(
                {
                    "gears_index": gene_index,
                    "original_gene_name": str(symbol),
                    "original_var_id": "" if ensembl_id is None else str(ensembl_id),
                    "resolved_symbol": resolved_symbol or "",
                    "scfoundation_gene_name": (
                        self.canonical_genes[token_index]
                        if token_index is not None
                        else ""
                    ),
                    "scfoundation_index": token_index,
                    "mapping_method": method,
                    "mapping_status": status,
                }
            )

        by_token = defaultdict(list)
        for row in provisional:
            if row["scfoundation_index"] is not None:
                by_token[int(row["scfoundation_index"])].append(row)

        collision_count = 0
        for token_index, rows in by_token.items():
            if len(rows) <= 1:
                continue
            rows.sort(
                key=lambda row: (
                    self._METHOD_PRIORITY.get(row["mapping_method"], 99),
                    int(row["gears_index"]),
                )
            )
            for row in rows[1:]:
                collision_count += 1
                row["mapping_status"] = "collision_fallback"
                row["mapping_method"] = row["mapping_method"] + ":collision"
                row["scfoundation_index"] = None
                row["scfoundation_gene_name"] = ""

        self.gene_mapping_rows = provisional
        self.gene_token_ids = [row["scfoundation_index"] for row in provisional]

        self.pert_mapping_rows = []
        self.pert_token_ids = []
        for pert_index, symbol in enumerate(self.pert_list):
            token_index, method, resolved_symbol, status = self._resolve_gene(symbol)
            self.pert_token_ids.append(token_index)
            self.pert_mapping_rows.append(
                {
                    "pert_index": pert_index,
                    "original_pert_name": str(symbol),
                    "resolved_symbol": resolved_symbol or "",
                    "scfoundation_gene_name": (
                        self.canonical_genes[token_index]
                        if token_index is not None
                        else ""
                    ),
                    "scfoundation_index": token_index,
                    "mapping_method": method,
                    "mapping_status": status,
                }
            )

        required_names = (
            self.required_perturbation_genes
            if self.required_perturbation_genes is not None
            else self.pert_list
        )
        self.required_pert_mapping_rows = []
        for required_index, symbol in enumerate(required_names):
            token_index, method, resolved_symbol, status = self._resolve_gene(symbol)
            self.required_pert_mapping_rows.append(
                {
                    "required_pert_index": required_index,
                    "original_pert_name": str(symbol),
                    "resolved_symbol": resolved_symbol or "",
                    "scfoundation_gene_name": (
                        self.canonical_genes[token_index]
                        if token_index is not None
                        else ""
                    ),
                    "scfoundation_index": token_index,
                    "mapping_method": method,
                    "mapping_status": status,
                }
            )

        matched_gene_pairs = [
            (index, int(token_id))
            for index, token_id in enumerate(self.gene_token_ids)
            if token_id is not None
        ]
        if not matched_gene_pairs:
            raise ValueError("No GEARS genes matched scFoundation's canonical genes.")

        self._matched_gene_indices = torch.tensor(
            [pair[0] for pair in matched_gene_pairs], dtype=torch.long
        )
        self._matched_token_indices = torch.tensor(
            [pair[1] for pair in matched_gene_pairs], dtype=torch.long
        )

        gene_matched = len(matched_gene_pairs)
        pert_matched = sum(token_id is not None for token_id in self.pert_token_ids)
        required_pert_matched = sum(
            row["scfoundation_index"] is not None
            for row in self.required_pert_mapping_rows
        )
        method_counts = Counter(
            row["mapping_method"]
            for row in self.gene_mapping_rows
            if row["mapping_status"] == "matched"
        )
        status_counts = Counter(row["mapping_status"] for row in self.gene_mapping_rows)
        pert_status_counts = Counter(
            row["mapping_status"] for row in self.pert_mapping_rows
        )
        self.mapping_summary = {
            "gene_count": len(self.gene_list),
            "gene_matched": gene_matched,
            "gene_missing": len(self.gene_list) - gene_matched,
            "gene_coverage": gene_matched / len(self.gene_list),
            "perturbation_count": len(self.pert_list),
            "perturbation_matched": pert_matched,
            "perturbation_missing": len(self.pert_list) - pert_matched,
            "perturbation_coverage": pert_matched / len(self.pert_list),
            "perturbation_graph_count": len(self.pert_list),
            "perturbation_graph_matched": pert_matched,
            "perturbation_graph_missing": len(self.pert_list) - pert_matched,
            "perturbation_graph_coverage": pert_matched / len(self.pert_list),
            "dataset_perturbation_count": len(required_names),
            "dataset_perturbation_matched": required_pert_matched,
            "dataset_perturbation_missing": (
                len(required_names) - required_pert_matched
            ),
            "dataset_perturbation_coverage": (
                required_pert_matched / len(required_names)
                if required_names
                else 1.0
            ),
            "collision_fallback_count": collision_count,
            "gene_method_counts": dict(sorted(method_counts.items())),
            "gene_status_counts": dict(sorted(status_counts.items())),
            "perturbation_status_counts": dict(sorted(pert_status_counts.items())),
            "canonical_gene_count": self.canonical_gene_count,
            "hgnc_path": str(self.hgnc_mapping_path),
            "hgnc_sha256": self.hgnc_sha256,
        }

        self._static_aligned_embeddings_cpu = None
        self._static_aligned_embeddings_device = None
        self._missing_fill_cpu = None
        self._write_mapping_reports()

        print(
            "[ScFoundationAdapter] GEARS gene coverage: "
            f"{gene_matched}/{len(self.gene_list)} = "
            f"{gene_matched / len(self.gene_list):.4f}"
        )
        print(
            "[ScFoundationAdapter] GEARS perturbation-graph coverage: "
            f"{pert_matched}/{len(self.pert_list)} = "
            f"{pert_matched / len(self.pert_list):.4f}"
        )
        print(
            "[ScFoundationAdapter] dataset perturbation-gene coverage: "
            f"{required_pert_matched}/{len(required_names)} = "
            f"{required_pert_matched / len(required_names):.4f}"
            if required_names
            else "[ScFoundationAdapter] dataset perturbation-gene coverage: 0/0"
        )
        print(
            "[ScFoundationAdapter] mapping methods: "
            + json.dumps(dict(sorted(method_counts.items())), ensure_ascii=False)
        )
        if collision_count:
            print(
                "[ScFoundationAdapter] duplicate-token collisions sent to fallback: "
                f"{collision_count}"
            )
        if pert_matched != len(self.pert_list):
            missing = [
                row["original_pert_name"]
                for row in self.pert_mapping_rows
                if row["mapping_status"] != "matched"
            ]
            print(
                "[ScFoundationAdapter] unmatched perturbation-graph nodes use "
                "fallback: "
                + ", ".join(missing)
            )
        required_missing = [
            row["original_pert_name"]
            for row in self.required_pert_mapping_rows
            if row["mapping_status"] != "matched"
        ]
        if required_missing and self.require_all_perturbations:
            raise ValueError(
                "Not all dataset perturbation genes map to scFoundation. See "
                "scfoundation_dataset_pert_mapping.tsv. Missing: "
                + ", ".join(required_missing)
            )

    def _write_mapping_reports(self):
        if self.mapping_report_dir is None:
            return
        self.mapping_report_dir.mkdir(parents=True, exist_ok=True)

        gene_path = self.mapping_report_dir / "scfoundation_gene_mapping.tsv"
        gene_fields = [
            "gears_index",
            "original_gene_name",
            "original_var_id",
            "resolved_symbol",
            "scfoundation_gene_name",
            "scfoundation_index",
            "mapping_method",
            "mapping_status",
        ]
        with gene_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=gene_fields, delimiter="\t")
            writer.writeheader()
            for row in self.gene_mapping_rows:
                output = dict(row)
                if output["scfoundation_index"] is None:
                    output["scfoundation_index"] = ""
                writer.writerow(output)

        pert_path = self.mapping_report_dir / "scfoundation_pert_mapping.tsv"
        pert_fields = [
            "pert_index",
            "original_pert_name",
            "resolved_symbol",
            "scfoundation_gene_name",
            "scfoundation_index",
            "mapping_method",
            "mapping_status",
        ]
        with pert_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pert_fields, delimiter="\t")
            writer.writeheader()
            for row in self.pert_mapping_rows:
                output = dict(row)
                if output["scfoundation_index"] is None:
                    output["scfoundation_index"] = ""
                writer.writerow(output)

        required_pert_path = (
            self.mapping_report_dir / "scfoundation_dataset_pert_mapping.tsv"
        )
        required_pert_fields = [
            "required_pert_index",
            "original_pert_name",
            "resolved_symbol",
            "scfoundation_gene_name",
            "scfoundation_index",
            "mapping_method",
            "mapping_status",
        ]
        with required_pert_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=required_pert_fields, delimiter="\t"
            )
            writer.writeheader()
            for row in self.required_pert_mapping_rows:
                output = dict(row)
                if output["scfoundation_index"] is None:
                    output["scfoundation_index"] = ""
                writer.writerow(output)

        summary_path = (
            self.mapping_report_dir / "scfoundation_gene_mapping_summary.json"
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.mapping_summary, handle, indent=2, ensure_ascii=False)

    def _import_official_loader(self):
        model_code = str(self.model_code_dir.resolve())
        if model_code not in sys.path:
            sys.path.insert(0, model_code)

        load_path = self.model_code_dir / "load.py"
        module_name = "_gears_scfoundation_official_load"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing

        spec = importlib.util.spec_from_file_location(module_name, load_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import scFoundation loader: {load_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == "local_attention":
                raise ModuleNotFoundError(
                    "scFoundation requires local-attention. Install the pinned "
                    "dependency with: python -m pip install local-attention==1.11.2"
                ) from exc
            raise
        return module

    @staticmethod
    def _torch_load_full(path):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    def _ensure_model_loaded(self):
        if self.scfoundation_model is not None:
            return
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device={self.device}, but CUDA is unavailable."
            )

        official = self._import_official_loader()
        checkpoint = self._torch_load_full(self.ckpt_path)
        if not isinstance(checkpoint, dict) or self.checkpoint_key not in checkpoint:
            keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
            raise KeyError(
                f"Checkpoint key '{self.checkpoint_key}' not found. Available: {keys}"
            )

        converted = official.convertconfig(checkpoint[self.checkpoint_key])
        config = dict(converted["config"])
        if "ppi_edge" not in config:
            config["ppi_edge"] = None
        config["device"] = self.device

        if config.get("model") != "mae_autobin" and config.get(
            "model_type"
        ) != "mae_autobin":
            raise ValueError(
                "Expected scFoundation mae_autobin checkpoint, got "
                f"model={config.get('model')}, model_type={config.get('model_type')}"
            )

        model = official.select_model(config)
        state = converted["model_state_dict"]
        model.load_state_dict(state, strict=True)

        if self.verify_loaded_tensors:
            loaded_state = model.state_dict()
            failed = []
            for key, checkpoint_tensor in state.items():
                actual = loaded_state[key].detach().cpu()
                expected = checkpoint_tensor.detach().cpu().to(actual.dtype)
                if not torch.equal(actual, expected):
                    failed.append(key)
            if failed:
                raise RuntimeError(
                    "Exact tensor verification failed for scFoundation keys: "
                    + ", ".join(failed[:50])
                )

        encoder = config.get("encoder", {})
        decoder = config.get("decoder", {})
        self.encoder_hidden_dim = int(encoder.get("hidden_dim", 0))
        self.decoder_hidden_dim = int(decoder.get("hidden_dim", 0))
        self.seq_len = int(config.get("seq_len", 0))
        self.pad_token_id = int(config.get("pad_token_id"))
        self.mask_token_id = int(config.get("mask_token_id"))
        self.bin_num = int(config.get("bin_num"))

        if self.seq_len != self.canonical_gene_count + 2:
            raise ValueError(
                f"Expected seq_len={self.canonical_gene_count + 2}, got {self.seq_len}."
            )
        if self.decoder_hidden_dim != self.gears_hidden_size:
            raise ValueError(
                "GEARS hidden_size must match scFoundation decoder hidden_dim: "
                f"{self.gears_hidden_size} vs {self.decoder_hidden_dim}."
            )
        if tuple(model.pos_emb.weight.shape) != (
            self.seq_len + 1,
            self.encoder_hidden_dim,
        ):
            raise ValueError(
                "Unexpected scFoundation pos_emb shape: "
                f"{tuple(model.pos_emb.weight.shape)}"
            )

        model.to_final = None
        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.scfoundation_model = model
        self.model_config = config
        self.loaded_tensor_count = len(state)
        del checkpoint, converted, state

        self._build_static_canonical_embeddings()
        print(
            "[ScFoundationAdapter] checkpoint loading verification passed: "
            f"{self.loaded_tensor_count} tensors loaded strictly; "
            f"encoder_dim={self.encoder_hidden_dim}, "
            f"decoder_dim={self.decoder_hidden_dim}, bin_num={self.bin_num}."
        )

    def _build_static_canonical_embeddings(self):
        if self._static_canonical_embeddings_cpu is not None:
            return
        model = self.scfoundation_model
        with torch.inference_mode():
            positional = model.pos_emb.weight[: self.canonical_gene_count]
            static = model.decoder_embed(positional)
            static = model.norm(static)
        self._static_canonical_embeddings_cpu = static.detach().float().cpu()
        if self.missing_strategy == "zero":
            self._missing_fill_cpu = torch.zeros(
                self.gears_hidden_size, dtype=torch.float32
            )
        else:
            self._missing_fill_cpu = (
                self._static_canonical_embeddings_cpu.mean(dim=0).detach()
            )

    def _aligned_static_embeddings(self):
        self._ensure_model_loaded()
        if self._static_aligned_embeddings_cpu is None:
            rows = []
            for token_id in self.gene_token_ids:
                if token_id is None:
                    rows.append(self._missing_fill_cpu)
                else:
                    rows.append(self._static_canonical_embeddings_cpu[int(token_id)])
            self._static_aligned_embeddings_cpu = torch.stack(rows, dim=0).detach()
        return self._static_aligned_embeddings_cpu

    def get_static_gene_embeddings(self, gene_list: List[str]):
        if self.gene_token_ids is None:
            raise RuntimeError("Call setup() before requesting scFoundation embeddings.")
        if list(gene_list) == self.gene_list:
            return self._aligned_static_embeddings().clone()

        self._ensure_model_loaded()
        token_ids = [self._resolve_gene(name)[0] for name in gene_list]
        rows = [
            self._missing_fill_cpu
            if token_id is None
            else self._static_canonical_embeddings_cpu[int(token_id)]
            for token_id in token_ids
        ]
        return torch.stack(rows, dim=0).clone()

    def get_pert_embeddings(self, pert_list: List[str]):
        if self.pert_token_ids is None:
            raise RuntimeError("Call setup() before requesting perturbation embeddings.")
        self._ensure_model_loaded()
        token_ids = [self._resolve_gene(name)[0] for name in pert_list]
        rows = [
            self._missing_fill_cpu
            if token_id is None
            else self._static_canonical_embeddings_cpu[int(token_id)]
            for token_id in token_ids
        ]
        return torch.stack(rows, dim=0).clone()

    def _build_contextual_fallback(self, batch_size):
        if self.contextual_fallback == "zero":
            return torch.zeros(
                batch_size,
                len(self.gene_list),
                self.gears_hidden_size,
                dtype=torch.float32,
                device=self.device,
            )
        if self._static_aligned_embeddings_device is None:
            self._static_aligned_embeddings_device = (
                self._aligned_static_embeddings()
                .to(self.device, dtype=torch.float32)
                .detach()
            )
        return (
            self._static_aligned_embeddings_device.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
            .clone()
        )

    def _autocast_settings(self):
        enabled = self.device.startswith("cuda") and self.precision != "fp32"
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return enabled, dtype

    def _pack_encoder_input(self, full_input):
        labels = full_input > 0
        counts = labels.sum(dim=1)



        packed_length = int(counts.max().item())
        if packed_length <= 0:
            raise ValueError("scFoundation encoder input contains no positive token.")

        encoder_values = torch.full(
            (full_input.shape[0], packed_length),
            float(self.pad_token_id),
            dtype=full_input.dtype,
            device=full_input.device,
        )
        encoder_positions = torch.full(
            (full_input.shape[0], packed_length),
            int(self.seq_len),
            dtype=torch.long,
            device=full_input.device,
        )
        encoder_padding = torch.ones(
            (full_input.shape[0], packed_length),
            dtype=torch.bool,
            device=full_input.device,
        )

        for row_index in range(full_input.shape[0]):
            positions = torch.nonzero(labels[row_index], as_tuple=False).flatten()
            length = int(positions.numel())
            if length:
                encoder_values[row_index, :length] = full_input[
                    row_index
                ].index_select(0, positions)
                encoder_positions[row_index, :length] = positions
                encoder_padding[row_index, :length] = False
        return encoder_values, encoder_positions, encoder_padding, labels

    def _forward_contextual_chunk(self, x_chunk):
        batch_size = int(x_chunk.shape[0])
        aligned = torch.zeros(
            batch_size,
            self.canonical_gene_count,
            dtype=torch.float32,
            device=self.device,
        )
        gene_indices = self._matched_gene_indices.to(self.device)
        token_indices = self._matched_token_indices.to(self.device)
        matched_values = x_chunk.index_select(1, gene_indices)
        aligned[:, token_indices] = matched_values

        aligned_sum = aligned.sum(dim=1)
        source_token = torch.log10(torch.clamp(aligned_sum, min=1e-8))
        target_token = torch.full_like(
            source_token, self.target_high_resolution
        )
        full_input = torch.cat(
            [aligned, target_token.unsqueeze(1), source_token.unsqueeze(1)], dim=1
        )

        (
            encoder_values,
            encoder_positions,
            encoder_padding,
            encoder_labels,
        ) = self._pack_encoder_input(full_input)
        decoder_positions = torch.arange(
            self.seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        decoder_padding = torch.zeros(
            full_input.shape, dtype=torch.bool, device=self.device
        )

        autocast_enabled, autocast_dtype = self._autocast_settings()
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda" if self.device.startswith("cuda") else "cpu",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                output = self.scfoundation_model.forward(
                    x=encoder_values,
                    padding_label=encoder_padding,
                    encoder_position_gene_ids=encoder_positions,
                    encoder_labels=encoder_labels,
                    decoder_data=full_input,
                    mask_gene_name=False,
                    mask_labels=None,
                    decoder_position_gene_ids=decoder_positions,
                    decoder_data_padding_labels=decoder_padding,
                )
        output = output[:, : self.canonical_gene_count, :]
        return output.index_select(1, token_indices).detach().float()

    def get_contextual_gene_embeddings(self, x: torch.Tensor, gene_list: List[str]):
        if self.gene_token_ids is None:
            raise RuntimeError("Call setup() before requesting contextual embeddings.")
        if list(gene_list) != self.gene_list:
            raise ValueError(
                "Contextual gene order differs from the order supplied to setup()."
            )
        if x.ndim != 2 or x.shape[1] != len(self.gene_list):
            raise ValueError(
                "Expected x with shape [batch, num_genes] = "
                f"[batch, {len(self.gene_list)}], got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("scFoundation input contains NaN or infinity.")
        if torch.any(x < 0):
            raise ValueError(
                "scFoundation single-cell expression input must be non-negative."
            )

        self._ensure_model_loaded()
        x = x.detach().to(self.device, dtype=torch.float32)
        batch_size = int(x.shape[0])
        full_embeddings = self._build_contextual_fallback(batch_size)
        gene_indices_device = self._matched_gene_indices.to(self.device)

        chunk_size = self.contextual_encoder_batch_size
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            contextual = self._forward_contextual_chunk(x[start:end])
            full_embeddings[start:end].index_copy_(
                1, gene_indices_device, contextual
            )
            del contextual
        return full_embeddings

    def get_config(self):
        return {
            "adapter_name": self.name,
            "source_dir": str(self.source_dir),
            "model_code_dir": str(self.model_code_dir),
            "ckpt_path": str(self.ckpt_path),
            "canonical_gene_path": str(self.canonical_gene_path),
            "hgnc_mapping_path": str(self.hgnc_mapping_path),
            "hgnc_sha256": self.hgnc_sha256,
            "expected_hgnc_sha256": self.expected_hgnc_sha256,
            "checkpoint_key": self.checkpoint_key,
            "device": self.device,
            "output_dim": self.output_dim,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "seq_len": self.seq_len,
            "bin_num": self.bin_num,
            "pre_normalized": self.pre_normalized,
            "target_high_resolution": self.target_high_resolution,
            "contextual_fallback": self.contextual_fallback,
            "contextual_encoder_batch_size": self.contextual_encoder_batch_size,
            "missing_strategy": self.missing_strategy,
            "precision": self.precision,
            "verify_loaded_tensors": self.verify_loaded_tensors,
            "require_all_perturbations": self.require_all_perturbations,
            "required_perturbation_genes": self.required_perturbation_genes,
            "loaded_tensor_count": self.loaded_tensor_count,
            "mapping_report_dir": (
                str(self.mapping_report_dir)
                if self.mapping_report_dir is not None
                else None
            ),
            "mapping_summary": self.mapping_summary,
            "seed": self.seed,
        }
