from .base import BaseSCFMAdapter


class NativeGEARSAdapter(BaseSCFMAdapter):
    name = "native_gears"
    output_dim = None

    def setup(self, gene_list, pert_list):
        self.gene_list = gene_list
        self.pert_list = pert_list

    def get_static_gene_embeddings(self, gene_list):
        return None

    def get_contextual_gene_embeddings(self, x, gene_list):
        return None

    def get_pert_embeddings(self, pert_list):
        return None
