# Gene-Perturbation Prediction

This directory contains GEARS-based gene-perturbation prediction workflows for
scKITE, scGPT, and scFoundation. The bundled GEARS-derived code is based on the
[Stanford SNAP GEARS repository](https://github.com/snap-stanford/GEARS).
The exact upstream commit and local modifications still need to be recorded
before public release.

## Structure

```text
gene_perturbation_prediction/
|-- adapters/                         # Model-specific GEARS adapters
|-- gears/                            # GEARS-derived implementation
|-- train_sckite.py
|-- train_scgpt.py
|-- train_scfoundation.py
|-- run_sckite_gears_norman.sh
|-- run_sckite_gears_replogle_exp7.sh
|-- run_scgpt_gears.sh
`-- run_scfoundation_gears.sh
```

The training scripts remain separate because the three backbones have different
checkpoint formats, preprocessing requirements, and external dependencies.

## scKITE runs

The scKITE launchers use repository-relative defaults for:

- `checkpoints/stage2/best.pt`;
- `vocab/global_vocab/gene_table.jsonl`;
- `sckite/stage2/model.py`;
- `data/gene_perturbation_prediction/`; and
- `outputs/gene_perturbation_prediction/`.

Run from the repository root:

```bash
bash downstream/gene_perturbation_prediction/run_sckite_gears_norman.sh
bash downstream/gene_perturbation_prediction/run_sckite_gears_replogle_exp7.sh
```

The defaults can be overridden with `PYTHON_BIN`, `DATA_ROOT`, `SAVE_ROOT`,
`SCKITE_CHECKPOINT_PATH`, `SCKITE_VOCAB_PATH`, `SCKITE_MODEL_PATH`, and
`SCKITE_MODEL_KWARGS_JSON`.

The scKITE command-line interface uses `adapter_name=sckite`. Gene embeddings
can be selected with `sckite_static` or `sckite_contextual`, and perturbation
embeddings can optionally be initialized with `sckite_init`.

For a deterministic random-embedding control, use `adapter_name=random` with
one of the scKITE embedding modes. The random adapter is seeded through the
main training `--seed` argument.

## Baseline runs

The baseline launchers require external source code and checkpoints that are
not distributed in this repository.

For scGPT, set:

```bash
SCGPT_SOURCE_DIR=/path/to/scgpt \
SCGPT_MODEL_DIR=/path/to/scgpt_checkpoint \
bash downstream/gene_perturbation_prediction/run_scgpt_gears.sh
```

For scFoundation, set:

```bash
SCFOUNDATION_SOURCE_DIR=/path/to/scfoundation \
SCFOUNDATION_CKPT=/path/to/models.ckpt \
bash downstream/gene_perturbation_prediction/run_scfoundation_gears.sh
```

Processed GEARS datasets are expected below
`data/gene_perturbation_prediction/<dataset>/`. Training results and logs are
written below `outputs/gene_perturbation_prediction/`.
