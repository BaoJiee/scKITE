# Cell-Type Annotation

[`cell_type_annotation.ipynb`](cell_type_annotation.ipynb) evaluates scKITE
cell representations for cell-type annotation. The notebook supports:

- frozen-encoder zero-shot annotation with a 5-nearest-neighbor classifier;
- supervised linear probing;
- supervised full fine-tuning.

The default zero-shot protocol uses the existing training split as the labeled
reference and the test split as the query set. Validation data are excluded
from the reference set unless explicitly enabled.

## Expected inputs

The released benchmark inputs are available from
[`CellType_Annotation`](https://huggingface.co/datasets/Baojiecau/Benchamark/tree/main/CellType_Annotation),
and the released Stage 2 checkpoint is available from
[`Baojiecau/scKITE`](https://huggingface.co/Baojiecau/scKITE/resolve/main/stage2_geosketch_50_20260814_085748/best.pt).

```text
data/cell_type_annotation/<dataset>/
|-- train/
|-- val/
`-- test/

checkpoints/stage2/best.pt
```

Edit the `CFG` dataclass near the beginning of the notebook to choose the data
root, checkpoint, evaluation mode, and plotting options. The benchmark does not
use a separate YAML file.

Generated embeddings, predictions, metrics, checkpoints, and figures are
written under `outputs/cell_type_annotation/`.
