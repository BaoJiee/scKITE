import importlib

import pytest


torch = pytest.importorskip("torch")
stage1_module = importlib.import_module("sckite.stage1.model")
stage2_module = importlib.import_module("sckite.stage2.model")
ScKITEStage1Model = stage1_module.ScKITEStage1Model
ScKITEStage2Model = stage2_module.ScKITEStage2Model
load_stage1_encoder_weights = stage2_module.load_stage1_encoder_weights


def build_stage1_model():
    return ScKITEStage1Model(
        global_vocab_size=32,
        d_model=8,
        n_heads=2,
        n_layers=1,
        expansion_ratio=2,
        dropout=0.0,
        value_hidden_dim=4,
        value_head_hidden_dim=8,
    )


def build_stage2_model():
    return ScKITEStage2Model(
        global_vocab_size=32,
        d_model=8,
        n_heads=2,
        n_layers=1,
        decoder_n_layers=1,
        decoder_n_heads=2,
        expansion_ratio=2,
        dropout=0.0,
        max_decoder_length=8,
        value_hidden_dim=4,
        value_head_hidden_dim=8,
    )


def test_stage1_forward_shapes():
    model = build_stage1_model().eval()
    gene_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    values = torch.tensor([[-1.0, 0.5, -3.0, -2.0], [-1.0, 1.0, 0.0, 2.0]])

    with torch.no_grad():
        output = model(
            encoder_input_gene_ids=gene_ids,
            encoder_input_values=values,
        )

    assert output["encoder_outputs"].shape == (2, 4, 8)
    assert output["cell_emb"].shape == (2, 8)
    assert output["expr_preds"].shape == (2, 4)


def test_stage2_routes_both_decoder_tasks():
    model = build_stage2_model().eval()
    gene_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    values = torch.tensor([[-1.0, 0.5, -3.0, -2.0], [-1.0, 1.0, 0.0, 2.0]])
    decoder_ids = torch.tensor([[1, 7, 8], [1, 9, 10]])
    decoder_mask = torch.ones_like(decoder_ids)
    task_ids = torch.tensor([0, 1])

    with torch.no_grad():
        output = model(
            encoder_input_gene_ids=gene_ids,
            encoder_input_values=values,
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_mask,
            decoder_task_ids=task_ids,
        )

    assert output["expr_preds"].shape == (2, 4)
    assert output["regulon_logits"].shape == (1, 3, 32)
    assert output["annotation_logits"].shape == (1, 3, 32)
    assert output["regulon_batch_indices"].tolist() == [0]
    assert output["annotation_batch_indices"].tolist() == [1]


def test_stage1_checkpoint_initializes_stage2_encoder(tmp_path):
    stage1_model = build_stage1_model()
    stage2_model = build_stage2_model()
    checkpoint_path = tmp_path / "stage1.pt"
    torch.save({"model_state_dict": stage1_model.state_dict()}, checkpoint_path)

    load_stage1_encoder_weights(
        stage2_model,
        checkpoint_path,
        strict_encoder=True,
        verbose=False,
    )

    assert torch.equal(
        stage2_model.shared_token_embedding.weight,
        stage1_model.shared_token_embedding.weight,
    )
    for target, source in zip(
        stage2_model.encoder.parameters(),
        stage1_model.encoder.parameters(),
    ):
        assert torch.equal(target, source)
