# scKITE

[![arXiv](https://img.shields.io/badge/arXiv-2609.14970-b31b1b.svg)](https://arxiv.org/abs/2609.14970)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20%26%20Data-FFD21E)](https://huggingface.co/Baojiecau/scKITE)

scKITE (**single-cell Knowledge-Integrated Transformer**) is a single-cell
foundation model (scFM) for learning and evaluating representations of
single-cell transcriptomic data.

This is the official implementation accompanying the preprint
[Towards a knowledge-enhanced single-cell foundation model](https://arxiv.org/abs/2609.14970).

**We present scKITE, a simple yet effective scFM that integrates
cell-annotation and gene-regulatory supervision into a shared transcriptomic
Transformer encoder through lightweight auxiliary decoders.**

The repository currently contains the Stage 1 encoder-pretraining workflow,
the Stage 2 multi-task workflow, vocabulary and regulon resources, downstream
benchmarks, and biological interpretation analyses used in the project.

The public release is intended to support:

- model training and checkpoint-based inference;
- construction and validation of a joint gene/text vocabulary;
- zero-shot batch-integration evaluation;
- cell-type annotation;
- gene-perturbation prediction with GEARS-based adapters; and
- attention-based biological interpretation analyses.

## Project status

The repository is under active release preparation. In particular:

- the dependency manifests are preliminary and still require validation against
  the final training environment;
- the Stage 2 checkpoint and selected pretraining and benchmark datasets are
  available on Hugging Face, while the Stage 1 checkpoint is not yet released;
- complete third-party provenance is still being consolidated; and
- end-to-end reproducibility tests are still being consolidated.

Scientific code is being reorganized without changing model behavior or the
experimental definitions used in the study.

## Model and analysis workflow

```text
Single-cell expression data
          |
          v
  Stage 1 encoder pretraining
          |
          v
  Stage 2 multi-task training
    |-- expression objective
    |-- regulon objective
    `-- annotation objective
          |
          v
  Downstream evaluation
    |-- batch integration
    |-- cell-type annotation
    `-- perturbation prediction
          |
          v
  Biological interpretation
```

Dataset construction, final hyperparameter selection, checkpoint provenance,
and complete reproduction commands will be documented before the first public
release.

See the [scKITE model and training guide](sckite/README.md) for the model input,
architecture, representations, objectives, and two-stage training procedure.

## Current repository layout

```text
.
|-- pyproject.toml                  # Package metadata and dependency groups
|-- environment.yml                # Reference Python and CUDA environment
|-- sckite/
|   |-- stage1/                    # Stage 1 encoder pretraining
|   `-- stage2/                    # Stage 2 knowledge-enhanced pretraining
|-- downstream/
|   |-- batch_integration/         # Notebook benchmark and helper module
|   |-- cell_type_annotation/      # Notebook benchmark
|   `-- gene_perturbation_prediction/
|-- biological_interpretation/     # Attention-based analyses
`-- vocab/                         # Vocabulary and regulon resources
```

### Core model

The [`sckite/stage1/`](sckite/stage1/) directory contains the encoder-only masked-expression
pretraining workflow:

- `model.py`: encoder and masked-expression value head;
- `data.py`: streaming-data utilities and encoder-only collation;
- `tokenizer.py`: the joint global gene/text tokenizer;
- `train.py`: distributed Stage 1 training and validation;
- `config.yaml`: the current Stage 1 configuration; and
- `run.sh`: the Stage 1 training launcher.

The [`sckite/stage2/`](sckite/stage2/) directory contains the Stage 2 workflow:

- `model.py`: model architecture and loss functions;
- `data.py`: streaming-data utilities and Stage 2 collation;
- `tokenizer.py`: the joint global gene/text tokenizer;
- `train.py`: distributed Stage 2 training and validation;
- `config.yaml`: the current Stage 2 configuration; and
- `run.sh`: the current server-oriented training launcher.

### Vocabulary and biological resources

The [`vocab/`](vocab/) directory contains:

- a global gene/text token mapping;
- metadata describing special-token and gene-token ranges;
- a gene table indexed by the global vocabulary; and
- a regulon-to-target-gene resource used by the Stage 2 regulon task.

Resource provenance, versions, licenses, and checksums are being compiled. Do
not redistribute these resources independently until their source licenses have
been documented.

### Downstream tasks

See the [downstream task guide](downstream/README.md) for inputs, entry points,
outputs, and task-specific requirements.

| Task | Current location | Current interface |
| --- | --- | --- |
| Batch integration | [`downstream/batch_integration/`](downstream/batch_integration/) | Notebook and helper module |
| Cell-type annotation | [`downstream/cell_type_annotation/`](downstream/cell_type_annotation/) | Notebook |
| Gene-perturbation prediction | [`downstream/gene_perturbation_prediction/`](downstream/gene_perturbation_prediction/) | Python and shell scripts |
| Biological interpretation | [`biological_interpretation/`](biological_interpretation/) | Analysis notebooks |

## Installation

Create the reference Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate sckite
```

For an existing Python 3.11 environment, install the core training package or
include the optional downstream, notebook, and tracking dependencies:

```bash
python -m pip install -e .
python -m pip install -e ".[downstream,notebooks,tracking]"
```

The current reference environment is:

- Python 3.11.14;
- PyTorch 2.10.0+cu128;
- CUDA 12.8; and
- NVIDIA RTX PRO 6000 GPU.

These values describe the environment used during current development; they do
not yet define the minimum GPU memory or hardware requirements. The CUDA wheel
source is recorded in `environment.yml`, while package groups and supported
Python versions are defined in `pyproject.toml`.

The current code imports the following major packages:

- PyTorch and PyTorch Geometric;
- Transformers;
- MosaicML Streaming;
- NumPy, pandas, SciPy, and scikit-learn;
- Scanpy and UMAP;
- Matplotlib and seaborn; and
- Weights & Biases (optional experiment tracking).

The non-core dependency lower bounds are an initial compatibility specification.
They should be validated and frozen from the final training environment before
the public release.

## Data

Raw and processed single-cell datasets are not committed to this repository.
The released training and benchmark inputs are hosted on Hugging Face:

| Resource | Hugging Face location | Contents |
| --- | --- | --- |
| Pretraining data | [50% pretraining corpus](https://huggingface.co/datasets/Baojiecau/pretrain_data/tree/main/50%25) | MosaicML Streaming `train/` and `val/` splits |
| Batch integration | [Batch_integration](https://huggingface.co/datasets/Baojiecau/Benchamark/tree/main/Batch_integration) | Benchmark datasets and split metadata |
| Cell-type annotation | [CellType_Annotation](https://huggingface.co/datasets/Baojiecau/Benchamark/tree/main/CellType_Annotation) | Benchmark datasets and split metadata |
| Gene-perturbation prediction | [Gene_Perturbation_Prediction](https://huggingface.co/datasets/Baojiecau/Benchamark/tree/main/Gene_Perturbation_Prediction) | Processed GEARS datasets, graph files, and splits |

Large files remain outside Git. Download the required directory from Hugging
Face and update the repository-relative paths in the corresponding YAML file or
notebook configuration cell.

For example, with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
hf download Baojiecau/pretrain_data \
  --repo-type dataset \
  --include "50%/*" \
  --local-dir data/pretrain_data

hf download Baojiecau/Benchamark \
  --repo-type dataset \
  --local-dir data/benchmarks
```

The following provenance information is still being consolidated for every
dataset:

1. the original source and citation;
2. the accession or persistent download URL;
3. the license and redistribution status;
4. preprocessing and train/validation/test split procedures; and
5. a checksum or immutable revision for the processed input.

## Model checkpoints

The released Stage 2 checkpoint is hosted in the
[scKITE Hugging Face model repository](https://huggingface.co/Baojiecau/scKITE):

| Checkpoint | Direct download | Size | SHA-256 |
| --- | --- | ---: | --- |
| Stage 2, 50% geosketch run | [`best.pt`](https://huggingface.co/Baojiecau/scKITE/resolve/main/stage2_geosketch_50_20260814_085748/best.pt) | 2,355,226,205 bytes | `5e0a09d4dc90a92676df06f6306e2bc9b9b1025ceec41f8229056da63ba6495a` |

Download it to the default downstream location with:

```bash
mkdir -p checkpoints/stage2
curl -L \
  https://huggingface.co/Baojiecau/scKITE/resolve/main/stage2_geosketch_50_20260814_085748/best.pt \
  -o checkpoints/stage2/best.pt
```

The Stage 1 initialization checkpoint, the exact vocabulary/resource revision
associated with the released checkpoint, and external baseline checkpoints are
not yet published here.

## Running the code

Stage 1 and Stage 2 use matching directory layouts and stage-local configuration
files. Update the data, tokenizer, and checkpoint locations in the relevant
`config.yaml`, then run:

```bash
bash sckite/stage1/run.sh
bash sckite/stage2/run.sh
```

After downloading the released artifacts above, set their local paths in the
relevant configuration file or notebook. External baseline locations still
need to be supplied separately. A verified minimal example and expected output
will be added before release.

## Testing

Install the development dependencies and run the test suite from the repository
root:

```bash
python -m pip install -e ".[dev]"
pytest
```

The initial tests validate Stage 1 and Stage 2 forward passes, Stage 1-to-Stage
2 checkpoint initialization, deterministic random-adapter behavior, and global
vocabulary consistency.

## Reproducibility checklist

- [ ] Pin Python and package versions.
- [ ] Document CUDA and hardware requirements.
- [ ] Document and validate the Stage 1 training workflow.
- [ ] Replace machine-specific paths with configuration parameters.
- [ ] Publish dataset manifests and preprocessing commands.
- [x] Publish the Stage 2 checkpoint location and checksum.
- [ ] Publish the Stage 1 checkpoint and a complete artifact manifest.
- [ ] Add a minimal end-to-end example.
- [ ] Add unit and integration tests.
- [ ] Add continuous integration.
- [ ] Record random seeds and expected benchmark outputs.
- [ ] Document third-party code and licenses.

## Citation

If you use scKITE, please cite:

> Hanqing Zhang, Jie Bao, Mei Ma, Shuai Liu, Jiaying Ma, Jiaguan Liu,
> Jiaxiao Li, Zhenbo Li, Wenwen Gong, and Zhijun Ca.
> *Towards a knowledge-enhanced single-cell foundation model*.
> arXiv:2609.14970, 2026.

```bibtex
@misc{zhang2026sckite,
  title={Towards a knowledge-enhanced single-cell foundation model},
  author={Zhang, Hanqing and Bao, Jie and Ma, Mei and Liu, Shuai and
          Ma, Jiaying and Liu, Jiaguan and Li, Jiaxiao and Li, Zhenbo and
          Gong, Wenwen and Ca, Zhijun},
  year={2026},
  eprint={2609.14970},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2609.14970}
}
```

Machine-readable citation metadata are available in [`CITATION.cff`](CITATION.cff).

## Third-party software

The perturbation-prediction workflow contains GEARS-derived or GEARS-compatible
code based on the
[Stanford SNAP GEARS repository](https://github.com/snap-stanford/GEARS), as
well as adapters for scGPT and scFoundation. Attribution, licensing, and the
current provenance limitation are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Original scKITE source code is released under the [MIT License](LICENSE).

The source-code license does not automatically apply to model weights,
datasets, pretrained tokenizers, third-party code, or biological resources.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for scope and attribution.

## Contact

For code-related questions and reproducibility reports, open a
[GitHub issue](https://github.com/BaoJiee/scKITE/issues).
