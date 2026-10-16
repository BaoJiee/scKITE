# Downstream Tasks

This directory contains the downstream evaluation workflows used with scKITE
representations. Each task remains self-contained because its data format,
evaluation protocol, and external dependencies differ.

| Task | Entry point | Purpose |
| --- | --- | --- |
| Batch integration | [`batch_integration/batch_integration.ipynb`](batch_integration/batch_integration.ipynb) | Frozen-encoder integration metrics |
| Cell-type annotation | [`cell_type_annotation/cell_type_annotation.ipynb`](cell_type_annotation/cell_type_annotation.ipynb) | Zero-shot k-NN and supervised annotation |
| Gene-perturbation prediction | [`gene_perturbation_prediction/`](gene_perturbation_prediction/) | GEARS-based perturbation prediction |

The batch-integration and cell-type-annotation benchmarks are notebook-driven
and do not use separate YAML configuration files. Their editable parameters are
kept in the configuration cell near the beginning of each notebook.

Task inputs should be placed under `data/<task>/`, checkpoints under
`checkpoints/`, and generated results under `outputs/<task>/`. These directories
are ignored by Git.

## Released benchmark data

| Task | Hugging Face data |
| --- | --- |
| Batch integration | [`Batch_integration`](https://huggingface.co/datasets/Baojiecau/Benchamark/tree/main/Batch_integration) |
| Cell-type annotation | [`CellType_Annotation`](https://huggingface.co/datasets/Baojiecau/Benchamark/tree/main/CellType_Annotation) |
| Gene-perturbation prediction | [`Gene_Perturbation_Prediction`](https://huggingface.co/datasets/Baojiecau/Benchamark/tree/main/Gene_Perturbation_Prediction) |

The released Stage 2 checkpoint is available from
[`Baojiecau/scKITE`](https://huggingface.co/Baojiecau/scKITE/resolve/main/stage2_geosketch_50_20260814_085748/best.pt).
Download the required artifacts outside Git and set their local paths in the
task notebook or launcher.

See each task directory for its required files and execution instructions.
