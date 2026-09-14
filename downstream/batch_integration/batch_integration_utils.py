from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm.auto import tqdm

try:
    from streaming import StreamingDataset
except ImportError:
    from streaming import Dataset as StreamingDataset


@dataclass
class MDSBatchIntegrationConfig:
    split_root: str
    output_root: str
    dataset_name: str = "Perirhinal_Cortex_MDS"
    train_dir_name: str = "train"
    val_dir_name: str = "val"
    test_dir_name: str = "test"
    gene_key: str = "genes"
    expr_key: str = "expressions"
    cell_type_key: str = "cell_type"
    batch_key: str = "batch"
    cell_id_key: str = "cell_id"
    max_length: Optional[int] = 1200
    add_cls_token: bool = True
    cls_token_id: int = 1
    pad_token_id: int = 0
    cls_expr_value: float = -1.0
    pad_expr_value: float = -2.0
    batch_size_train: int = 32
    batch_size_eval: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_train: bool = False
    shuffle_train: bool = True
    seed: int = 42
    save_metadata: bool = True


@dataclass
class MDSBatchIntegrationContext:
    dataset_name: str
    output_root: Path
    meta_df: pd.DataFrame
    cell_type_to_id: Dict[str, int]
    id_to_cell_type: Dict[int, str]
    batch_to_id: Dict[str, int]
    id_to_batch: Dict[int, str]
    cell_type_ids: np.ndarray
    batch_ids: np.ndarray
    split_labels: np.ndarray
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    all_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    embed_loader_all: DataLoader
    metadata: Dict[str, Any] = field(default_factory=dict)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_scalar_value(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return str(x.item())
    return str(x)


def convert_to_1d_numpy(x: Any, dtype: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, bytes):
        text = x.decode("utf-8")
        arr = np.asarray(json.loads(text))
    elif isinstance(x, str):
        text = x.strip()
        if text.startswith("[") and text.endswith("]"):
            arr = np.asarray(json.loads(text))
        else:
            arr = np.asarray([x])
    else:
        arr = np.asarray(x)
    arr = np.ravel(arr)
    arr = arr.astype(dtype)
    return arr


def read_mds_column_info(mds_dir: str) -> Dict[str, Any]:
    index_path = Path(mds_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到index.json：{index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index_info = json.load(f)
    shard0 = index_info["shards"][0]
    column_names = shard0["column_names"]
    column_encodings = shard0["column_encodings"]
    columns = dict(zip(column_names, column_encodings))
    return {"index_info": index_info, "columns": columns}


def check_split_dirs(cfg: MDSBatchIntegrationConfig) -> Dict[str, Path]:
    split_root = Path(cfg.split_root)
    split_dirs = {
        "train": split_root / cfg.train_dir_name,
        "val": split_root / cfg.val_dir_name,
        "test": split_root / cfg.test_dir_name,
    }
    for split_name, split_dir in split_dirs.items():
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_name} MDS目录不存在：{split_dir}")
        if not (split_dir / "index.json").exists():
            raise FileNotFoundError(f"{split_name} MDS缺少index.json：{split_dir / 'index.json'}")
    return split_dirs


def scan_one_split_metadata(mds_dir: str, split_name: str, cfg: MDSBatchIntegrationConfig) -> pd.DataFrame:
    ds = StreamingDataset(local=str(mds_dir), shuffle=False)
    rows = []
    for local_idx in tqdm(range(len(ds)), desc=f"Scan metadata: {split_name}"):
        sample = ds[local_idx]
        cell_type = decode_scalar_value(sample[cfg.cell_type_key])
        batch = decode_scalar_value(sample[cfg.batch_key])
        if cfg.cell_id_key in sample:
            cell_id = decode_scalar_value(sample[cfg.cell_id_key])
        else:
            cell_id = f"{split_name}_{local_idx}"
        rows.append({
            "split": split_name,
            "local_idx": int(local_idx),
            "cell_id": cell_id,
            "cell_type": cell_type,
            "batch": batch,
        })
    meta_df = pd.DataFrame(rows)
    return meta_df


def scan_all_split_metadata(cfg: MDSBatchIntegrationConfig) -> pd.DataFrame:
    split_dirs = check_split_dirs(cfg)
    meta_parts = []
    for split_name in ["train", "val", "test"]:
        one_meta = scan_one_split_metadata(str(split_dirs[split_name]), split_name, cfg)
        meta_parts.append(one_meta)
    meta_df = pd.concat(meta_parts, axis=0, ignore_index=True)
    meta_df["global_idx"] = np.arange(meta_df.shape[0], dtype=np.int64)
    return meta_df


class SplitMDSCellDataset(Dataset):
    def __init__(self, mds_dir: str, split_name: str, cfg: MDSBatchIntegrationConfig, cell_type_to_id: Dict[str, int], batch_to_id: Dict[str, int], global_offset: int = 0) -> None:
        self.mds_dir = str(mds_dir)
        self.split_name = str(split_name)
        self.cfg = cfg
        self.cell_type_to_id = cell_type_to_id
        self.batch_to_id = batch_to_id
        self.global_offset = int(global_offset)
        self.ds = StreamingDataset(local=self.mds_dir, shuffle=False)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, local_idx: int) -> Dict[str, Any]:
        sample = self.ds[int(local_idx)]
        genes = convert_to_1d_numpy(sample[self.cfg.gene_key], dtype=np.int64)
        exprs = convert_to_1d_numpy(sample[self.cfg.expr_key], dtype=np.float32)
        if genes.shape[0] != exprs.shape[0]:
            raise ValueError(f"{self.split_name}[{local_idx}] genes长度{genes.shape[0]}与expressions长度{exprs.shape[0]}不一致。")
        max_gene_len = self.cfg.max_length - 1 if self.cfg.add_cls_token and self.cfg.max_length is not None else self.cfg.max_length
        if max_gene_len is not None and genes.shape[0] > max_gene_len:
            genes = genes[:max_gene_len]
            exprs = exprs[:max_gene_len]
        if self.cfg.add_cls_token:
            genes = np.concatenate([[self.cfg.cls_token_id], genes]).astype(np.int64)
            exprs = np.concatenate([[self.cfg.cls_expr_value], exprs]).astype(np.float32)
        cell_type = decode_scalar_value(sample[self.cfg.cell_type_key])
        batch = decode_scalar_value(sample[self.cfg.batch_key])
        if self.cfg.cell_id_key in sample:
            cell_id = decode_scalar_value(sample[self.cfg.cell_id_key])
        else:
            cell_id = f"{self.split_name}_{local_idx}"
        item = {
            "genes": genes,
            "expressions": exprs,
            "cell_type_id": int(self.cell_type_to_id[cell_type]),
            "batch_id": int(self.batch_to_id[batch]),
            "cell_type": cell_type,
            "batch": batch,
            "cell_id": cell_id,
            "split": self.split_name,
            "local_idx": int(local_idx),
            "row_idx": int(self.global_offset + int(local_idx)),
        }
        return item


def collate_mds_batch(batch: List[Dict[str, Any]], cfg: MDSBatchIntegrationConfig) -> Dict[str, Any]:
    lengths = [len(x["genes"]) for x in batch]
    max_len = max(lengths)
    if cfg.max_length is not None:
        max_len = min(max_len, cfg.max_length)
    batch_size = len(batch)
    gene_tensor = torch.full((batch_size, max_len), int(cfg.pad_token_id), dtype=torch.long)
    expr_tensor = torch.full((batch_size, max_len), float(cfg.pad_expr_value), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    cell_type_ids = torch.empty(batch_size, dtype=torch.long)
    batch_ids = torch.empty(batch_size, dtype=torch.long)
    row_idx = torch.empty(batch_size, dtype=torch.long)
    cell_ids = []
    split_names = []
    for i, item in enumerate(batch):
        genes = torch.as_tensor(item["genes"][:max_len], dtype=torch.long)
        exprs = torch.as_tensor(item["expressions"][:max_len], dtype=torch.float32)
        cur_len = genes.numel()
        gene_tensor[i, :cur_len] = genes
        expr_tensor[i, :cur_len] = exprs
        padding_mask[i, :cur_len] = False
        cell_type_ids[i] = int(item["cell_type_id"])
        batch_ids[i] = int(item["batch_id"])
        row_idx[i] = int(item["row_idx"])
        cell_ids.append(item["cell_id"])
        split_names.append(item["split"])
    out = {
        "genes": gene_tensor,
        "gene_ids": gene_tensor,
        "expressions": expr_tensor,
        "expr_values": expr_tensor,
        "padding_mask": padding_mask,
        "attention_mask": (~padding_mask).long(),
        "cell_type_id": cell_type_ids,
        "cell_type_ids": cell_type_ids,
        "batch_id": batch_ids,
        "batch_ids": batch_ids,
        "row_idx": row_idx,
        "cell_id": cell_ids,
        "split": split_names,
    }
    return out


def make_cell_type_and_batch_maps(meta_df: pd.DataFrame) -> Dict[str, Any]:
    cell_types = sorted(meta_df["cell_type"].astype(str).unique().tolist())
    batches = sorted(meta_df["batch"].astype(str).unique().tolist())
    cell_type_to_id = {name: idx for idx, name in enumerate(cell_types)}
    batch_to_id = {name: idx for idx, name in enumerate(batches)}
    id_to_cell_type = {idx: name for name, idx in cell_type_to_id.items()}
    id_to_batch = {idx: name for name, idx in batch_to_id.items()}
    return {
        "cell_type_to_id": cell_type_to_id,
        "batch_to_id": batch_to_id,
        "id_to_cell_type": id_to_cell_type,
        "id_to_batch": id_to_batch,
    }


def save_interface_metadata(meta_df: pd.DataFrame, mapping_dict: Dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(output_root / "batch_integration_mds_metadata.csv", index=False)
    pd.Series(mapping_dict["cell_type_to_id"], name="cell_type_id").rename_axis("cell_type").reset_index().to_csv(output_root / "cell_type_to_id.csv", index=False)
    pd.Series(mapping_dict["batch_to_id"], name="batch_id").rename_axis("batch").reset_index().to_csv(output_root / "batch_to_id.csv", index=False)
    pd.crosstab(meta_df["cell_type"], meta_df["split"]).to_csv(output_root / "cell_type_by_split_from_interface.csv")
    pd.crosstab(meta_df["batch"], meta_df["split"]).to_csv(output_root / "batch_by_split_from_interface.csv")


def prepare_split_mds_for_batch_integration(cfg: MDSBatchIntegrationConfig) -> MDSBatchIntegrationContext:
    set_global_seed(cfg.seed)
    split_dirs = check_split_dirs(cfg)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    column_info = read_mds_column_info(str(split_dirs["train"]))
    columns = column_info["columns"]
    required_cols = [cfg.gene_key, cfg.expr_key, cfg.cell_type_key, cfg.batch_key]
    for col in required_cols:
        if col not in columns:
            raise KeyError(f"train MDS中缺少必要列：{col}；当前列为：{list(columns.keys())}")
    meta_df = scan_all_split_metadata(cfg)
    mapping_dict = make_cell_type_and_batch_maps(meta_df)
    cell_type_to_id = mapping_dict["cell_type_to_id"]
    batch_to_id = mapping_dict["batch_to_id"]
    id_to_cell_type = mapping_dict["id_to_cell_type"]
    id_to_batch = mapping_dict["id_to_batch"]
    meta_df["cell_type_id"] = meta_df["cell_type"].map(cell_type_to_id).astype(int)
    meta_df["batch_id"] = meta_df["batch"].map(batch_to_id).astype(int)
    train_n = int((meta_df["split"] == "train").sum())
    val_n = int((meta_df["split"] == "val").sum())
    test_n = int((meta_df["split"] == "test").sum())
    train_dataset = SplitMDSCellDataset(str(split_dirs["train"]), "train", cfg, cell_type_to_id, batch_to_id, global_offset=0)
    val_dataset = SplitMDSCellDataset(str(split_dirs["val"]), "val", cfg, cell_type_to_id, batch_to_id, global_offset=train_n)
    test_dataset = SplitMDSCellDataset(str(split_dirs["test"]), "test", cfg, cell_type_to_id, batch_to_id, global_offset=train_n + val_n)
    all_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    collate_fn = lambda batch: collate_mds_batch(batch, cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_train, shuffle=cfg.shuffle_train, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=cfg.drop_last_train, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False, collate_fn=collate_fn)
    embed_loader_all = DataLoader(all_dataset, batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False, collate_fn=collate_fn)
    cell_type_ids = meta_df["cell_type_id"].values.astype(np.int64)
    batch_ids = meta_df["batch_id"].values.astype(np.int64)
    split_labels = meta_df["split"].values.astype(str)
    if cfg.save_metadata:
        save_interface_metadata(meta_df, mapping_dict, output_root)
    context = MDSBatchIntegrationContext(
        dataset_name=cfg.dataset_name,
        output_root=output_root,
        meta_df=meta_df,
        cell_type_to_id=cell_type_to_id,
        id_to_cell_type=id_to_cell_type,
        batch_to_id=batch_to_id,
        id_to_batch=id_to_batch,
        cell_type_ids=cell_type_ids,
        batch_ids=batch_ids,
        split_labels=split_labels,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        all_dataset=all_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        embed_loader_all=embed_loader_all,
        metadata={
            "columns": columns,
            "train_dir": str(split_dirs["train"]),
            "val_dir": str(split_dirs["val"]),
            "test_dir": str(split_dirs["test"]),
            "cfg": cfg,
            "embed_loader_all": embed_loader_all,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        },
    )
    print("MDS batch integration接口准备完成。")
    print(f"dataset: {cfg.dataset_name}")
    print(f"n_cells: {meta_df.shape[0]}")
    print(f"n_train: {train_n}")
    print(f"n_val: {val_n}")
    print(f"n_test: {test_n}")
    print(f"n_cell_type: {len(cell_type_to_id)}")
    print(f"n_batch: {len(batch_to_id)}")
    return context


def inspect_one_batch(context: MDSBatchIntegrationContext, split: str = "train") -> Dict[str, Any]:
    loader_map = {"train": context.train_loader, "val": context.val_loader, "test": context.test_loader, "all": context.embed_loader_all}
    if split not in loader_map:
        raise ValueError(f"split必须是{list(loader_map.keys())}之一，但收到：{split}")
    one_batch = next(iter(loader_map[split]))
    print("genes:", one_batch["genes"].shape, one_batch["genes"].dtype)
    print("expressions:", one_batch["expressions"].shape, one_batch["expressions"].dtype)
    print("padding_mask:", one_batch["padding_mask"].shape, one_batch["padding_mask"].dtype)
    print("cell_type_id:", one_batch["cell_type_id"].shape, one_batch["cell_type_id"].dtype)
    print("batch_id:", one_batch["batch_id"].shape, one_batch["batch_id"].dtype)
    print("row_idx:", one_batch["row_idx"][:10])
    print("split:", one_batch["split"][:5])
    return one_batch
