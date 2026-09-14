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
