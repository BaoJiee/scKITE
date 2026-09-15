import math
from typing import List, Optional

import torch

from .base import BaseSCFMAdapter


class RandomAdapter(BaseSCFMAdapter):
    name = "random"

    def __init__(
        self,
        output_dim: Optional[int] = None,
        gears_hidden_size: Optional[int] = None,
        seed: int = 1,
        device: str = "cuda",
        cache_dir=None,
        **kwargs,
    ):
        resolved_dim = output_dim if output_dim is not None else gears_hidden_size
        if resolved_dim is None or int(resolved_dim) <= 0:
            raise ValueError("output_dim or gears_hidden_size must be a positive integer.")

        self.output_dim = int(resolved_dim)
        self.seed = int(seed)
        self._gene_embeddings = None
        self._pert_embeddings = None
        super().__init__(
            device=device,
            cache_dir=cache_dir,
            output_dim=self.output_dim,
            seed=self.seed,
            **kwargs,
        )

    def setup(
        self,
        gene_list: List[str],
        pert_list: List[str],
        gene_id_list=None,
    ):
        self.gene_list = list(gene_list)
        self.pert_list = list(pert_list)
        self.gene_id_list = None if gene_id_list is None else list(gene_id_list)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        scale = 1.0 / math.sqrt(self.output_dim)
        self._gene_embeddings = (
            torch.randn(len(self.gene_list), self.output_dim, generator=generator) * scale
        )
        self._pert_embeddings = (
            torch.randn(len(self.pert_list), self.output_dim, generator=generator) * scale
        )

    def _require_setup(self):
        if self._gene_embeddings is None or self._pert_embeddings is None:
            raise RuntimeError("Call setup() before requesting random embeddings.")

    def get_static_gene_embeddings(self, gene_list: List[str]) -> torch.Tensor:
        self._require_setup()
        if list(gene_list) != self.gene_list:
            raise ValueError("gene_list must match the list passed to setup().")
        return self._gene_embeddings.clone()

    def get_contextual_gene_embeddings(
        self,
        x: torch.Tensor,
        gene_list: List[str],
    ) -> torch.Tensor:
        static_embeddings = self.get_static_gene_embeddings(gene_list)
        if x.ndim != 2 or x.shape[1] != len(self.gene_list):
            raise ValueError(
                f"x must have shape [batch_size, {len(self.gene_list)}], got {tuple(x.shape)}."
            )
        return static_embeddings.to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1).clone()

    def get_pert_embeddings(self, pert_list: List[str]) -> torch.Tensor:
        self._require_setup()
        if list(pert_list) != self.pert_list:
            raise ValueError("pert_list must match the list passed to setup().")
        return self._pert_embeddings.clone()
