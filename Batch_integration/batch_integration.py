from pathlib import Path  # 导入路径处理工具。 #
from dataclasses import dataclass, field  # 导入dataclass和field。 #
from typing import Any, Dict, List, Optional  # 导入类型注解。 #
import json  # 导入json模块。 #
import random  # 导入随机数模块。 #
import numpy as np  # 导入numpy模块。 #
import pandas as pd  # 导入pandas模块。 #
import torch  # 导入PyTorch模块。 #
from torch.utils.data import Dataset, DataLoader, ConcatDataset  # 导入Dataset、DataLoader和ConcatDataset。 #
from tqdm.auto import tqdm  # 导入进度条工具。 #

try:  # 尝试导入新版streaming接口。 #
    from streaming import StreamingDataset  # 导入StreamingDataset。 #
except ImportError:  # 如果新版接口不可用。 #
    from streaming import Dataset as StreamingDataset  # 使用旧版Dataset作为StreamingDataset。 #


@dataclass  # 定义MDS batch integration接口配置类。 #
class MDSBatchIntegrationConfig:  # 配置类开始。 #
    split_root: str  # 已经划分好的MDS根目录，里面应包含train、val、test。 #
    output_root: str  # batch integration任务输出目录。 #
    dataset_name: str = "Perirhinal_Cortex_MDS"  # 数据集名称。 #
    train_dir_name: str = "train"  # train目录名称。 #
    val_dir_name: str = "val"  # val目录名称。 #
    test_dir_name: str = "test"  # test目录名称。 #
    gene_key: str = "genes"  # MDS中基因token列名。 #
    expr_key: str = "expressions"  # MDS中表达值列名。 #
    cell_type_key: str = "cell_type"  # MDS中cell type列名。 #
    batch_key: str = "batch"  # MDS中batch列名。 #
    cell_id_key: str = "cell_id"  # MDS中cell id列名。 #
    max_length: Optional[int] = 1200  # 最大输入长度。 #
    add_cls_token: bool = True  # 是否在序列开头添加CLS token。 #
    cls_token_id: int = 1  # CLS token id。 #
    pad_token_id: int = 0  # PAD token id。 #
    cls_expr_value: float = -1.0  # CLS位置表达值。 #
    pad_expr_value: float = -2.0  # PAD位置表达值。 #
    batch_size_train: int = 32  # 训练集batch size。 #
    batch_size_eval: int = 64  # 验证、测试和全量embedding batch size。 #
    num_workers: int = 0  # DataLoader进程数。 #
    pin_memory: bool = True  # 是否启用pin_memory。 #
    drop_last_train: bool = False  # 训练集是否丢弃最后一个不完整batch。 #
    shuffle_train: bool = True  # 训练集是否shuffle。 #
    seed: int = 42  # 随机种子。 #
    save_metadata: bool = True  # 是否保存metadata和检查表。 #


@dataclass  # 定义MDS batch integration上下文类。 #
class MDSBatchIntegrationContext:  # 上下文类开始。 #
    dataset_name: str  # 数据集名称。 #
    output_root: Path  # 输出目录。 #
    meta_df: pd.DataFrame  # 全部细胞metadata表。 #
    cell_type_to_id: Dict[str, int]  # cell type到id映射。 #
    id_to_cell_type: Dict[int, str]  # id到cell type映射。 #
    batch_to_id: Dict[str, int]  # batch到id映射。 #
    id_to_batch: Dict[int, str]  # id到batch映射。 #
    cell_type_ids: np.ndarray  # 全部细胞的cell type id数组。 #
    batch_ids: np.ndarray  # 全部细胞的batch id数组。 #
    split_labels: np.ndarray  # 全部细胞的split标签数组。 #
    train_dataset: Dataset  # train dataset对象。 #
    val_dataset: Dataset  # val dataset对象。 #
    test_dataset: Dataset  # test dataset对象。 #
    all_dataset: Dataset  # train、val、test合并后的dataset对象。 #
    train_loader: DataLoader  # train DataLoader。 #
    val_loader: DataLoader  # val DataLoader。 #
    test_loader: DataLoader  # test DataLoader。 #
    embed_loader_all: DataLoader  # 全量embedding DataLoader。 #
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外metadata字典。 #


def set_global_seed(seed: int = 42) -> None:  # 定义函数：固定随机种子。 #
    random.seed(seed)  # 固定Python随机种子。 #
    np.random.seed(seed)  # 固定numpy随机种子。 #
    torch.manual_seed(seed)  # 固定PyTorch CPU随机种子。 #
    if torch.cuda.is_available():  # 判断CUDA是否可用。 #
        torch.cuda.manual_seed_all(seed)  # 固定PyTorch CUDA随机种子。 #


def decode_scalar_value(x: Any) -> str:  # 定义函数：把MDS中的标量值转成字符串。 #
    if isinstance(x, bytes):  # 判断是否为bytes类型。 #
        return x.decode("utf-8")  # 解码为字符串。 #
    if isinstance(x, np.bytes_):  # 判断是否为numpy bytes类型。 #
        return x.decode("utf-8")  # 解码为字符串。 #
    if isinstance(x, np.ndarray) and x.ndim == 0:  # 判断是否为0维numpy数组。 #
        return str(x.item())  # 取出标量并转成字符串。 #
    return str(x)  # 其他类型直接转成字符串。 #


def convert_to_1d_numpy(x: Any, dtype: Any) -> np.ndarray:  # 定义函数：把MDS字段转换为一维numpy数组。 #
    if isinstance(x, torch.Tensor):  # 判断是否为torch Tensor。 #
        arr = x.detach().cpu().numpy()  # 转换为numpy数组。 #
    elif isinstance(x, np.ndarray):  # 判断是否已经是numpy数组。 #
        arr = x  # 直接使用原数组。 #
    elif isinstance(x, bytes):  # 判断是否为bytes。 #
        text = x.decode("utf-8")  # 解码为字符串。 #
        arr = np.asarray(json.loads(text))  # 按json列表解析为数组。 #
    elif isinstance(x, str):  # 判断是否为字符串。 #
        text = x.strip()  # 去除首尾空白。 #
        if text.startswith("[") and text.endswith("]"):  # 判断是否像json列表。 #
            arr = np.asarray(json.loads(text))  # 按json列表解析为数组。 #
        else:  # 如果不是json列表。 #
            arr = np.asarray([x])  # 包装成单元素数组。 #
    else:  # 其他类型。 #
        arr = np.asarray(x)  # 尝试直接转换为numpy数组。 #
    arr = np.ravel(arr)  # 拉平成一维数组。 #
    arr = arr.astype(dtype)  # 转换为指定数据类型。 #
    return arr  # 返回一维数组。 #


def read_mds_column_info(mds_dir: str) -> Dict[str, Any]:  # 定义函数：读取MDS列信息。 #
    index_path = Path(mds_dir) / "index.json"  # 构造index.json路径。 #
    if not index_path.exists():  # 判断index.json是否存在。 #
        raise FileNotFoundError(f"未找到index.json：{index_path}")  # 抛出文件不存在错误。 #
    with index_path.open("r", encoding="utf-8") as f:  # 打开index.json文件。 #
        index_info = json.load(f)  # 读取json内容。 #
    shard0 = index_info["shards"][0]  # 读取第一个shard信息。 #
    column_names = shard0["column_names"]  # 读取列名。 #
    column_encodings = shard0["column_encodings"]  # 读取列编码。 #
    columns = dict(zip(column_names, column_encodings))  # 构造列名到编码的字典。 #
    return {"index_info": index_info, "columns": columns}  # 返回MDS列信息。 #


def check_split_dirs(cfg: MDSBatchIntegrationConfig) -> Dict[str, Path]:  # 定义函数：检查train、val、test目录。 #
    split_root = Path(cfg.split_root)  # 转换split根目录为Path对象。 #
    split_dirs = {  # 构造split目录字典。 #
        "train": split_root / cfg.train_dir_name,  # train目录路径。 #
        "val": split_root / cfg.val_dir_name,  # val目录路径。 #
        "test": split_root / cfg.test_dir_name,  # test目录路径。 #
    }  # split目录字典结束。 #
    for split_name, split_dir in split_dirs.items():  # 遍历三个split目录。 #
        if not split_dir.exists():  # 判断目录是否存在。 #
            raise FileNotFoundError(f"{split_name} MDS目录不存在：{split_dir}")  # 抛出错误。 #
        if not (split_dir / "index.json").exists():  # 判断index.json是否存在。 #
            raise FileNotFoundError(f"{split_name} MDS缺少index.json：{split_dir / 'index.json'}")  # 抛出错误。 #
    return split_dirs  # 返回split目录字典。 #


def scan_one_split_metadata(mds_dir: str, split_name: str, cfg: MDSBatchIntegrationConfig) -> pd.DataFrame:  # 定义函数：扫描一个split的metadata。 #
    ds = StreamingDataset(local=str(mds_dir), shuffle=False)  # 固定顺序读取MDS。 #
    rows = []  # 初始化metadata行列表。 #
    for local_idx in tqdm(range(len(ds)), desc=f"Scan metadata: {split_name}"):  # 遍历当前split所有样本。 #
        sample = ds[local_idx]  # 读取当前样本。 #
        cell_type = decode_scalar_value(sample[cfg.cell_type_key])  # 读取cell type。 #
        batch = decode_scalar_value(sample[cfg.batch_key])  # 读取batch。 #
        if cfg.cell_id_key in sample:  # 判断是否存在cell_id列。 #
            cell_id = decode_scalar_value(sample[cfg.cell_id_key])  # 读取cell_id。 #
        else:  # 如果没有cell_id列。 #
            cell_id = f"{split_name}_{local_idx}"  # 构造默认cell_id。 #
        rows.append({  # 添加一行metadata。 #
            "split": split_name,  # 保存split名称。 #
            "local_idx": int(local_idx),  # 保存当前split内部行号。 #
            "cell_id": cell_id,  # 保存cell_id。 #
            "cell_type": cell_type,  # 保存cell type。 #
            "batch": batch,  # 保存batch。 #
        })  # 当前样本metadata添加结束。 #
    meta_df = pd.DataFrame(rows)  # 转换为DataFrame。 #
    return meta_df  # 返回metadata表。 #


def scan_all_split_metadata(cfg: MDSBatchIntegrationConfig) -> pd.DataFrame:  # 定义函数：扫描三个split的metadata。 #
    split_dirs = check_split_dirs(cfg)  # 检查并获取split目录。 #
    meta_parts = []  # 初始化metadata列表。 #
    for split_name in ["train", "val", "test"]:  # 按固定顺序遍历split。 #
        one_meta = scan_one_split_metadata(str(split_dirs[split_name]), split_name, cfg)  # 扫描当前split。 #
        meta_parts.append(one_meta)  # 保存当前split metadata。 #
    meta_df = pd.concat(meta_parts, axis=0, ignore_index=True)  # 合并三个split的metadata。 #
    meta_df["global_idx"] = np.arange(meta_df.shape[0], dtype=np.int64)  # 构造全局行号。 #
    return meta_df  # 返回总metadata。 #


class SplitMDSCellDataset(Dataset):  # 定义单个split的MDS细胞数据集。 #
    def __init__(self, mds_dir: str, split_name: str, cfg: MDSBatchIntegrationConfig, cell_type_to_id: Dict[str, int], batch_to_id: Dict[str, int], global_offset: int = 0) -> None:  # 初始化函数。 #
        self.mds_dir = str(mds_dir)  # 保存MDS目录。 #
        self.split_name = str(split_name)  # 保存split名称。 #
        self.cfg = cfg  # 保存配置对象。 #
        self.cell_type_to_id = cell_type_to_id  # 保存cell type映射。 #
        self.batch_to_id = batch_to_id  # 保存batch映射。 #
        self.global_offset = int(global_offset)  # 保存全局行号偏移。 #
        self.ds = StreamingDataset(local=self.mds_dir, shuffle=False)  # 读取MDS数据集。 #

    def __len__(self) -> int:  # 定义返回样本数的函数。 #
        return len(self.ds)  # 返回MDS样本数。 #

    def __getitem__(self, local_idx: int) -> Dict[str, Any]:  # 定义取样函数。 #
        sample = self.ds[int(local_idx)]  # 读取当前样本。 #
        genes = convert_to_1d_numpy(sample[self.cfg.gene_key], dtype=np.int64)  # 读取基因token数组。 #
        exprs = convert_to_1d_numpy(sample[self.cfg.expr_key], dtype=np.float32)  # 读取表达量数组。 #
        if genes.shape[0] != exprs.shape[0]:  # 检查genes和expressions长度是否一致。 #
            raise ValueError(f"{self.split_name}[{local_idx}] genes长度{genes.shape[0]}与expressions长度{exprs.shape[0]}不一致。")  # 抛出错误。 #
        max_gene_len = self.cfg.max_length - 1 if self.cfg.add_cls_token and self.cfg.max_length is not None else self.cfg.max_length  # 计算不含CLS的最大基因长度。 #
        if max_gene_len is not None and genes.shape[0] > max_gene_len:  # 判断是否超过最大长度。 #
            genes = genes[:max_gene_len]  # 截断genes。 #
            exprs = exprs[:max_gene_len]  # 截断expressions。 #
        if self.cfg.add_cls_token:  # 判断是否需要添加CLS。 #
            genes = np.concatenate([[self.cfg.cls_token_id], genes]).astype(np.int64)  # 在开头添加CLS token。 #
            exprs = np.concatenate([[self.cfg.cls_expr_value], exprs]).astype(np.float32)  # 在开头添加CLS表达值。 #
        cell_type = decode_scalar_value(sample[self.cfg.cell_type_key])  # 读取cell type字符串。 #
        batch = decode_scalar_value(sample[self.cfg.batch_key])  # 读取batch字符串。 #
        if self.cfg.cell_id_key in sample:  # 判断是否存在cell_id列。 #
            cell_id = decode_scalar_value(sample[self.cfg.cell_id_key])  # 读取cell_id。 #
        else:  # 如果没有cell_id列。 #
            cell_id = f"{self.split_name}_{local_idx}"  # 构造默认cell_id。 #
        item = {  # 构造返回样本。 #
            "genes": genes,  # 保存genes数组。 #
            "expressions": exprs,  # 保存expressions数组。 #
            "cell_type_id": int(self.cell_type_to_id[cell_type]),  # 保存cell type id。 #
            "batch_id": int(self.batch_to_id[batch]),  # 保存batch id。 #
            "cell_type": cell_type,  # 保存cell type字符串。 #
            "batch": batch,  # 保存batch字符串。 #
            "cell_id": cell_id,  # 保存cell_id。 #
            "split": self.split_name,  # 保存split名称。 #
            "local_idx": int(local_idx),  # 保存局部行号。 #
            "row_idx": int(self.global_offset + int(local_idx)),  # 保存全局行号。 #
        }  # 返回样本构造结束。 #
        return item  # 返回样本。 #


def collate_mds_batch(batch: List[Dict[str, Any]], cfg: MDSBatchIntegrationConfig) -> Dict[str, Any]:  # 定义batch整理函数。 #
    lengths = [len(x["genes"]) for x in batch]  # 获取每个样本长度。 #
    max_len = max(lengths)  # 获取当前batch最大长度。 #
    if cfg.max_length is not None:  # 判断是否设置全局最大长度。 #
        max_len = min(max_len, cfg.max_length)  # 限制当前batch最大长度。 #
    batch_size = len(batch)  # 获取batch size。 #
    gene_tensor = torch.full((batch_size, max_len), int(cfg.pad_token_id), dtype=torch.long)  # 初始化gene tensor。 #
    expr_tensor = torch.full((batch_size, max_len), float(cfg.pad_expr_value), dtype=torch.float32)  # 初始化expression tensor。 #
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)  # 初始化padding mask，True表示padding。 #
    cell_type_ids = torch.empty(batch_size, dtype=torch.long)  # 初始化cell type id tensor。 #
    batch_ids = torch.empty(batch_size, dtype=torch.long)  # 初始化batch id tensor。 #
    row_idx = torch.empty(batch_size, dtype=torch.long)  # 初始化row_idx tensor。 #
    cell_ids = []  # 初始化cell_id列表。 #
    split_names = []  # 初始化split列表。 #
    for i, item in enumerate(batch):  # 遍历batch内样本。 #
        genes = torch.as_tensor(item["genes"][:max_len], dtype=torch.long)  # 转换genes为tensor。 #
        exprs = torch.as_tensor(item["expressions"][:max_len], dtype=torch.float32)  # 转换expressions为tensor。 #
        cur_len = genes.numel()  # 获取当前样本长度。 #
        gene_tensor[i, :cur_len] = genes  # 写入gene tensor。 #
        expr_tensor[i, :cur_len] = exprs  # 写入expression tensor。 #
        padding_mask[i, :cur_len] = False  # 标记真实token位置。 #
        cell_type_ids[i] = int(item["cell_type_id"])  # 写入cell type id。 #
        batch_ids[i] = int(item["batch_id"])  # 写入batch id。 #
        row_idx[i] = int(item["row_idx"])  # 写入全局row_idx。 #
        cell_ids.append(item["cell_id"])  # 保存cell_id。 #
        split_names.append(item["split"])  # 保存split名称。 #
    out = {  # 构造输出batch字典。 #
        "genes": gene_tensor,  # 输出genes。 #
        "gene_ids": gene_tensor,  # 输出gene_ids别名。 #
        "expressions": expr_tensor,  # 输出expressions。 #
        "expr_values": expr_tensor,  # 输出expr_values别名。 #
        "padding_mask": padding_mask,  # 输出padding mask。 #
        "attention_mask": (~padding_mask).long(),  # 输出attention mask，1表示真实token。 #
        "cell_type_id": cell_type_ids,  # 输出cell_type_id。 #
        "cell_type_ids": cell_type_ids,  # 输出cell_type_ids别名。 #
        "batch_id": batch_ids,  # 输出batch_id。 #
        "batch_ids": batch_ids,  # 输出batch_ids别名。 #
        "row_idx": row_idx,  # 输出row_idx。 #
        "cell_id": cell_ids,  # 输出cell_id。 #
        "split": split_names,  # 输出split。 #
    }  # 输出batch字典结束。 #
    return out  # 返回整理后的batch。 #


def make_cell_type_and_batch_maps(meta_df: pd.DataFrame) -> Dict[str, Any]:  # 定义函数：构建标签映射。 #
    cell_types = sorted(meta_df["cell_type"].astype(str).unique().tolist())  # 获取并排序所有cell type。 #
    batches = sorted(meta_df["batch"].astype(str).unique().tolist())  # 获取并排序所有batch。 #
    cell_type_to_id = {name: idx for idx, name in enumerate(cell_types)}  # 构建cell type到id映射。 #
    batch_to_id = {name: idx for idx, name in enumerate(batches)}  # 构建batch到id映射。 #
    id_to_cell_type = {idx: name for name, idx in cell_type_to_id.items()}  # 构建id到cell type映射。 #
    id_to_batch = {idx: name for name, idx in batch_to_id.items()}  # 构建id到batch映射。 #
    return {  # 返回映射字典。 #
        "cell_type_to_id": cell_type_to_id,  # 返回cell type到id映射。 #
        "batch_to_id": batch_to_id,  # 返回batch到id映射。 #
        "id_to_cell_type": id_to_cell_type,  # 返回id到cell type映射。 #
        "id_to_batch": id_to_batch,  # 返回id到batch映射。 #
    }  # 映射字典结束。 #


def save_interface_metadata(meta_df: pd.DataFrame, mapping_dict: Dict[str, Any], output_root: Path) -> None:  # 定义函数：保存metadata和检查表。 #
    output_root.mkdir(parents=True, exist_ok=True)  # 创建输出目录。 #
    meta_df.to_csv(output_root / "batch_integration_mds_metadata.csv", index=False)  # 保存总metadata表。 #
    pd.Series(mapping_dict["cell_type_to_id"], name="cell_type_id").rename_axis("cell_type").reset_index().to_csv(output_root / "cell_type_to_id.csv", index=False)  # 保存cell type映射。 #
    pd.Series(mapping_dict["batch_to_id"], name="batch_id").rename_axis("batch").reset_index().to_csv(output_root / "batch_to_id.csv", index=False)  # 保存batch映射。 #
    pd.crosstab(meta_df["cell_type"], meta_df["split"]).to_csv(output_root / "cell_type_by_split_from_interface.csv")  # 保存cell type分布检查表。 #
    pd.crosstab(meta_df["batch"], meta_df["split"]).to_csv(output_root / "batch_by_split_from_interface.csv")  # 保存batch分布检查表。 #


def prepare_split_mds_for_batch_integration(cfg: MDSBatchIntegrationConfig) -> MDSBatchIntegrationContext:  # 定义核心接口函数。 #
    set_global_seed(cfg.seed)  # 固定随机种子。 #
    split_dirs = check_split_dirs(cfg)  # 检查并获取split目录。 #
    output_root = Path(cfg.output_root)  # 转换输出目录为Path对象。 #
    output_root.mkdir(parents=True, exist_ok=True)  # 创建输出目录。 #
    column_info = read_mds_column_info(str(split_dirs["train"]))  # 读取train MDS列信息。 #
    columns = column_info["columns"]  # 获取列信息。 #
    required_cols = [cfg.gene_key, cfg.expr_key, cfg.cell_type_key, cfg.batch_key]  # 定义必要列。 #
    for col in required_cols:  # 遍历必要列。 #
        if col not in columns:  # 判断必要列是否存在。 #
            raise KeyError(f"train MDS中缺少必要列：{col}；当前列为：{list(columns.keys())}")  # 抛出错误。 #
    meta_df = scan_all_split_metadata(cfg)  # 扫描三个split的metadata。 #
    mapping_dict = make_cell_type_and_batch_maps(meta_df)  # 构建cell type和batch映射。 #
    cell_type_to_id = mapping_dict["cell_type_to_id"]  # 取出cell type到id映射。 #
    batch_to_id = mapping_dict["batch_to_id"]  # 取出batch到id映射。 #
    id_to_cell_type = mapping_dict["id_to_cell_type"]  # 取出id到cell type映射。 #
    id_to_batch = mapping_dict["id_to_batch"]  # 取出id到batch映射。 #
    meta_df["cell_type_id"] = meta_df["cell_type"].map(cell_type_to_id).astype(int)  # 添加cell type id列。 #
    meta_df["batch_id"] = meta_df["batch"].map(batch_to_id).astype(int)  # 添加batch id列。 #
    train_n = int((meta_df["split"] == "train").sum())  # 统计train细胞数。 #
    val_n = int((meta_df["split"] == "val").sum())  # 统计val细胞数。 #
    test_n = int((meta_df["split"] == "test").sum())  # 统计test细胞数。 #
    train_dataset = SplitMDSCellDataset(str(split_dirs["train"]), "train", cfg, cell_type_to_id, batch_to_id, global_offset=0)  # 构建train dataset。 #
    val_dataset = SplitMDSCellDataset(str(split_dirs["val"]), "val", cfg, cell_type_to_id, batch_to_id, global_offset=train_n)  # 构建val dataset。 #
    test_dataset = SplitMDSCellDataset(str(split_dirs["test"]), "test", cfg, cell_type_to_id, batch_to_id, global_offset=train_n + val_n)  # 构建test dataset。 #
    all_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])  # 合并三个dataset。 #
    collate_fn = lambda batch: collate_mds_batch(batch, cfg)  # 构造带cfg的collate函数。 #
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_train, shuffle=cfg.shuffle_train, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=cfg.drop_last_train, collate_fn=collate_fn)  # 构建train loader。 #
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False, collate_fn=collate_fn)  # 构建val loader。 #
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False, collate_fn=collate_fn)  # 构建test loader。 #
    embed_loader_all = DataLoader(all_dataset, batch_size=cfg.batch_size_eval, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False, collate_fn=collate_fn)  # 构建全量embedding loader。 #
    cell_type_ids = meta_df["cell_type_id"].values.astype(np.int64)  # 提取cell type id数组。 #
    batch_ids = meta_df["batch_id"].values.astype(np.int64)  # 提取batch id数组。 #
    split_labels = meta_df["split"].values.astype(str)  # 提取split标签数组。 #
    if cfg.save_metadata:  # 判断是否保存metadata。 #
        save_interface_metadata(meta_df, mapping_dict, output_root)  # 保存metadata和检查表。 #
    context = MDSBatchIntegrationContext(  # 构造context对象。 #
        dataset_name=cfg.dataset_name,  # 保存数据集名称。 #
        output_root=output_root,  # 保存输出目录。 #
        meta_df=meta_df,  # 保存metadata表。 #
        cell_type_to_id=cell_type_to_id,  # 保存cell type到id映射。 #
        id_to_cell_type=id_to_cell_type,  # 保存id到cell type映射。 #
        batch_to_id=batch_to_id,  # 保存batch到id映射。 #
        id_to_batch=id_to_batch,  # 保存id到batch映射。 #
        cell_type_ids=cell_type_ids,  # 保存cell type id数组。 #
        batch_ids=batch_ids,  # 保存batch id数组。 #
        split_labels=split_labels,  # 保存split标签数组。 #
        train_dataset=train_dataset,  # 保存train dataset。 #
        val_dataset=val_dataset,  # 保存val dataset。 #
        test_dataset=test_dataset,  # 保存test dataset。 #
        all_dataset=all_dataset,  # 保存全量dataset。 #
        train_loader=train_loader,  # 保存train loader。 #
        val_loader=val_loader,  # 保存val loader。 #
        test_loader=test_loader,  # 保存test loader。 #
        embed_loader_all=embed_loader_all,  # 保存全量embedding loader。 #
        metadata={  # 构造额外metadata字典。 #
            "columns": columns,  # 保存MDS列信息。 #
            "train_dir": str(split_dirs["train"]),  # 保存train路径。 #
            "val_dir": str(split_dirs["val"]),  # 保存val路径。 #
            "test_dir": str(split_dirs["test"]),  # 保存test路径。 #
            "cfg": cfg,  # 保存配置对象。 #
            "embed_loader_all": embed_loader_all,  # 保存全量embedding loader。 #
            "train_loader": train_loader,  # 保存train loader。 #
            "val_loader": val_loader,  # 保存val loader。 #
            "test_loader": test_loader,  # 保存test loader。 #
        },  # 额外metadata字典结束。 #
    )  # context对象构造结束。 #
    print("MDS batch integration接口准备完成。")  # 打印完成提示。 #
    print(f"dataset: {cfg.dataset_name}")  # 打印数据集名称。 #
    print(f"n_cells: {meta_df.shape[0]}")  # 打印总细胞数。 #
    print(f"n_train: {train_n}")  # 打印train细胞数。 #
    print(f"n_val: {val_n}")  # 打印val细胞数。 #
    print(f"n_test: {test_n}")  # 打印test细胞数。 #
    print(f"n_cell_type: {len(cell_type_to_id)}")  # 打印cell type数量。 #
    print(f"n_batch: {len(batch_to_id)}")  # 打印batch数量。 #
    return context  # 返回context对象。 #


def inspect_one_batch(context: MDSBatchIntegrationContext, split: str = "train") -> Dict[str, Any]:  # 定义函数：检查一个batch。 #
    loader_map = {"train": context.train_loader, "val": context.val_loader, "test": context.test_loader, "all": context.embed_loader_all}  # 构建loader字典。 #
    if split not in loader_map:  # 判断split是否合法。 #
        raise ValueError(f"split必须是{list(loader_map.keys())}之一，但收到：{split}")  # 抛出错误。 #
    one_batch = next(iter(loader_map[split]))  # 取出一个batch。 #
    print("genes:", one_batch["genes"].shape, one_batch["genes"].dtype)  # 打印genes形状。 #
    print("expressions:", one_batch["expressions"].shape, one_batch["expressions"].dtype)  # 打印expressions形状。 #
    print("padding_mask:", one_batch["padding_mask"].shape, one_batch["padding_mask"].dtype)  # 打印padding_mask形状。 #
    print("cell_type_id:", one_batch["cell_type_id"].shape, one_batch["cell_type_id"].dtype)  # 打印cell_type_id形状。 #
    print("batch_id:", one_batch["batch_id"].shape, one_batch["batch_id"].dtype)  # 打印batch_id形状。 #
    print("row_idx:", one_batch["row_idx"][:10])  # 打印前10个row_idx。 #
    print("split:", one_batch["split"][:5])  # 打印前5个split。 #
    return one_batch  # 返回这个batch。 #