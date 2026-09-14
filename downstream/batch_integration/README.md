# Batch Integration

[`batch_integration.ipynb`](batch_integration.ipynb) is the benchmark entry
point. It loads the encoder side of a scKITE checkpoint, freezes all model
parameters, extracts `<cls>` cell embeddings, and evaluates biological
conservation and batch-mixing metrics. Cell-type and batch labels are used only
for evaluation.

[`batch_integration_utils.py`](batch_integration_utils.py) provides the MDS
dataset reader, collation, metadata preparation, and split validation used by
the notebook. It is a helper module rather than a separate benchmark entry.

## Expected inputs

The default notebook configuration expects:

```text
data/batch_integration/<dataset>/
|-- train/
|-- val/
`-- test/

checkpoints/stage2/best.pt
sckite/stage2/model.py
```

Update `USER_CONFIG` in the notebook to select the dataset, checkpoint, batch
size, and evaluation settings. No benchmark YAML file is required.

## Outputs

Results are written to:

```text
outputs/batch_integration/<dataset>/
```

The notebook is stored without cached outputs so that repository copies do not
contain machine-specific paths or stale results.
