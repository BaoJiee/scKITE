import importlib

import pytest


torch = pytest.importorskip("torch")
adapters_module = importlib.import_module(
    "downstream.gene_perturbation_prediction.adapters"
)
RandomAdapter = adapters_module.RandomAdapter
build_adapter = adapters_module.build_adapter


def build_random_adapter():
    adapter = build_adapter("random", device="cpu", output_dim=4, seed=7)
    adapter.setup(
        gene_list=["GENE_A", "GENE_B"],
        pert_list=["GENE_A", "GENE_C"],
        gene_id_list=["ENSG_A", "ENSG_B"],
    )
    return adapter


def test_random_adapter_is_registered_and_deterministic():
    first = build_random_adapter()
    second = build_random_adapter()

    assert isinstance(first, RandomAdapter)
    assert torch.equal(
        first.get_static_gene_embeddings(first.gene_list),
        second.get_static_gene_embeddings(second.gene_list),
    )
    assert torch.equal(
        first.get_pert_embeddings(first.pert_list),
        second.get_pert_embeddings(second.pert_list),
    )


def test_random_adapter_embedding_shapes():
    adapter = build_random_adapter()
    expression = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    static = adapter.get_static_gene_embeddings(adapter.gene_list)
    contextual = adapter.get_contextual_gene_embeddings(expression, adapter.gene_list)
    perturbation = adapter.get_pert_embeddings(adapter.pert_list)

    assert static.shape == (2, 4)
    assert contextual.shape == (3, 2, 4)
    assert perturbation.shape == (2, 4)
