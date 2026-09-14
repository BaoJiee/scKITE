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

See each task directory for its required files and execution instructions.
