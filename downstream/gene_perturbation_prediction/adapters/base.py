from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch


class BaseSCFMAdapter(ABC):
    name = "base"
    output_dim = None

    def __init__(self, device="cuda", cache_dir=None, **kwargs):
        self.device = device
        self.cache_dir = cache_dir
        self.kwargs = kwargs
        self.gene_list = None
        self.pert_list = None

    @abstractmethod
    def setup(self, gene_list: List[str], pert_list: List[str]):
        raise NotImplementedError

    def get_static_gene_embeddings(self, gene_list: List[str]) -> Optional[torch.Tensor]:
        return None

    def get_contextual_gene_embeddings(self, x: torch.Tensor, gene_list: List[str]) -> Optional[torch.Tensor]:
        return None

    def get_pert_embeddings(self, pert_list: List[str]) -> Optional[torch.Tensor]:
        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.name,
            "output_dim": self.output_dim,
            "device": self.device,
            "cache_dir": self.cache_dir,
            **self.kwargs,
        }
