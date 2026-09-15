import json
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_vocabulary_metadata_uses_repository_relative_paths():
    metadata_path = REPOSITORY_ROOT / "vocab" / "global_vocab" / "global_vocab_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    for key in ("raw_gene_vocab_path", "text_tokenizer_path"):
        value = metadata[key]
        path = PurePosixPath(value)
        assert not path.is_absolute()
        assert ":" not in value
        assert ".." not in path.parts

    raw_gene_vocab_path = PurePosixPath(metadata["raw_gene_vocab_path"])
    assert (REPOSITORY_ROOT / Path(*raw_gene_vocab_path.parts)).is_file()


def test_special_token_ids_match_global_vocabulary():
    vocabulary_root = REPOSITORY_ROOT / "vocab" / "global_vocab"
    metadata = json.loads(
        (vocabulary_root / "global_vocab_meta.json").read_text(encoding="utf-8")
    )
    vocabulary = json.loads(
        (vocabulary_root / "global_vocab.json").read_text(encoding="utf-8")
    )

    assert max(vocabulary.values()) < metadata["vocab_size_for_embedding"]
    assert vocabulary[metadata["pad_token"]] == metadata["pad_token_id"]
    assert vocabulary[metadata["cls_token"]] == metadata["cls_token_id"]
    assert (
        vocabulary[metadata["regulon_sep_token"]]
        == metadata["regulon_sep_token_id"]
    )
    for token, token_id in metadata["special_token_ids"].items():
        assert vocabulary[token] == token_id


def test_gene_table_matches_global_vocabulary():
    vocabulary_root = REPOSITORY_ROOT / "vocab" / "global_vocab"
    vocabulary = json.loads(
        (vocabulary_root / "global_vocab.json").read_text(encoding="utf-8")
    )

    gene_count = 0
    with (vocabulary_root / "gene_table.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            assert vocabulary[record["global_token"]] == record["global_id"]
            gene_count += 1

    assert gene_count > 0
