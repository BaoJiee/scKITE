# scKITE Model and Training

This directory contains the two-stage training implementation for scKITE
(single-cell Knowledge-Integrated Transformer). Stage 1 and Stage 2 use the
same file layout so that model definitions, data pipelines, training code, and
configuration remain easy to compare.

## Directory layout

```text
sckite/
|-- stage1/
|   |-- model.py
|   |-- data.py
|   |-- tokenizer.py
|   |-- train.py
|   |-- config.yaml
|   `-- run.sh
`-- stage2/
    |-- model.py
    |-- data.py
    |-- tokenizer.py
    |-- train.py
    |-- config.yaml
    `-- run.sh
```

The public model classes are:

- `sckite.stage1.model.ScKITEStage1Model`
- `sckite.stage2.model.ScKITEStage2Model`

## Model overview

scKITE uses a shared transcriptomic Transformer encoder. The current reference
architecture has 12 pre-normalization Transformer layers, a hidden dimension of
512, eight attention heads, a feed-forward expansion ratio of four, and dropout
of 0.1.

Each cell is represented as a sequence of gene identities and expression
values. Positive expression values are represented using 50 non-zero quantile
levels, with zero reserved for non-expressed values. The maximum encoder length
is 2,049 tokens, including the leading `<cls>` token. Gene-token embeddings,
continuous expression embeddings, and the expression-mask indicator are added
before the sequence is passed to the encoder.

The final `<cls>` hidden state is used as the cell representation. Hidden states
at gene positions are used as contextual gene representations.

Input records must already contain the intended gene filtering and ordering.
The current collators map genes to global IDs and preserve the stored order;
they do not independently sort genes by expression.

## Stage 1

Stage 1 performs transcriptomic self-supervised pretraining using masked
expression reconstruction. Gene identities remain visible while 30% of valid
expression values are replaced by the mask value. A regression head predicts
the original expression levels using mean-squared error over masked positions.

Main files:

- [`stage1/model.py`](stage1/model.py): encoder-only model and value head.
- [`stage1/data.py`](stage1/data.py): Stage 1 collator and streaming loader.
- [`stage1/train.py`](stage1/train.py): distributed training and validation.
- [`stage1/config.yaml`](stage1/config.yaml): paths and hyperparameters.
- [`stage1/run.sh`](stage1/run.sh): training launcher.

The Stage 1 checkpoint initializes the Stage 2 encoder.

## Stage 2

Stage 2 continues masked-expression reconstruction with a masking probability
of 10% and adds two lightweight auxiliary decoders:

- a regulon decoder for profile-specific transcription-factor and target-gene
  sequences;
- an annotation decoder for paired natural-language descriptions.

Both decoders contain two autoregressive Transformer decoder layers with causal
self-attention and cross-attention over the shared encoder states. The decoders
have independent parameters and output heads. The current expression, regulon,
and annotation loss weights are all 1.0.

Main files:

- [`stage2/model.py`](stage2/model.py): shared encoder and auxiliary decoders.
- [`stage2/data.py`](stage2/data.py): mixed-task collation and data loading.
- [`stage2/train.py`](stage2/train.py): distributed multi-objective training.
- [`stage2/config.yaml`](stage2/config.yaml): paths and hyperparameters.
- [`stage2/run.sh`](stage2/run.sh): training launcher.

After Stage 2, downstream tasks use the encoder representations without
requiring annotation text or regulon targets.

## Required paths

The example configurations use repository-relative locations:

| Artifact | Default location |
| --- | --- |
| Stage 1 training data | `data/stage1/train` |
| Stage 1 validation data | `data/stage1/val` |
| Stage 2 training data | `data/stage2/train` |
| Stage 2 validation data | `data/stage2/val` |
| Biomedical tokenizer | `vocab/biobert_tokenizer` |
| Global vocabulary | `vocab/global_vocab/global_vocab.json` |
| Gene table | `vocab/global_vocab/gene_table.jsonl` |
| Regulon targets | `vocab/regulon_targets_importance_sorted.json` |
| Stage 1 initialization checkpoint | `checkpoints/stage1/best.pt` |
| Training outputs | `outputs/stage1` and `outputs/stage2` |

Data, tokenizer files, checkpoints, and outputs are not committed to Git unless
they are small versioned resources. Update the relevant `config.yaml` when a
different location is used.

## Running training

Run commands from the repository root:

```bash
bash sckite/stage1/run.sh
bash sckite/stage2/run.sh
```

Both launchers use the active Python environment. To activate a named Conda
environment inside the launcher, set `SCKITE_CONDA_ENV`:

```bash
SCKITE_CONDA_ENV=my_environment bash sckite/stage1/run.sh
```

The reference development environment is Python 3.11.14, PyTorch
2.10.0+cu128, CUDA 12.8, and an NVIDIA RTX PRO 6000 GPU. The initial package
specification is provided in [`../pyproject.toml`](../pyproject.toml), and the
reference Conda setup is provided in
[`../environment.yml`](../environment.yml). Exact optional-package versions
should be frozen from the final training environment before release.

## Checkpoints

Training checkpoints store model parameters as `model_state_dict` together
with optimizer, scheduler, scaler, epoch, and configuration state when
available. Renaming the Python model class does not change tensor keys in a
`state_dict` checkpoint.

The release locations and checksums for the final Stage 1 and Stage 2
checkpoints are not yet available.
