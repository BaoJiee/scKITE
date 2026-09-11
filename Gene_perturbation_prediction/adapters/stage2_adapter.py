import json  # 导入 json，用于读取 json/jsonl vocab。 #
from pathlib import Path  # 导入 Path，用于处理文件路径。 #
from typing import List, Optional  # 导入类型注释。 #
import importlib.util
import sys  # 用于从任意路径导入 Stage2 模型文件。 #

import torch  # 导入 PyTorch。 #

from .base import BaseSCFMAdapter

def left_binning(values: torch.Tensor, num_bins: int) -> torch.Tensor:  # 对表达值做左侧分桶。 #
    if values.numel() == 0:  # 如果没有元素。 #
        return values  # 直接返回。 #
    out = torch.zeros_like(values, dtype=torch.long)  # 初始化输出，默认 0 表示零表达桶。 #
    pos_mask = values > 0  # 找到正表达值位置。 #
    if pos_mask.sum() == 0:  # 如果没有正表达值。 #
        return out  # 返回全 0。 #
    pos_values = values[pos_mask].float()  # 取出正表达值。 #
    if pos_values.numel() == 1:  # 如果只有一个正值。 #
        out[pos_mask] = 1  # 直接分到第一个正桶。 #
        return out  # 返回结果。 #
    q = torch.linspace(0.0, 1.0, steps=int(num_bins), device=values.device)  # 构造分位数位置。 #
    edges = torch.quantile(pos_values, q)  # 计算分位数边界。 #
    inner_edges = edges[1:-1].contiguous()  # 去掉首尾边界。 #
    bucket_ids = torch.bucketize(pos_values, inner_edges, right=False) + 1  # 对正表达值分桶并从 1 开始。 #
    out[pos_mask] = bucket_ids.long()  # 写回正表达值位置。 #
    return out  # 返回分桶结果。 #
  # 导入 adapter 抽象基类。 #


class Stage2Adapter(BaseSCFMAdapter):  # 定义 Stage2 adapter。 #
    name = "stage2"  # adapter 名称。 #

    def __init__(  # 初始化 Stage2 adapter。 #
        self,  # 实例对象。 #
        ckpt_path,  # best.pt 路径。 #
        vocab_path,  # gene_vocabulary.jsonl 路径。 #
        device="cuda",  # 运行设备。 #
        gears_hidden_size=64,  # GEARS hidden_size。 #
        missing_strategy="mean_gene",  # 缺失基因填充策略。 #
        project_method="slice",  # 512 到 hidden_size 的投影方式。 #
        normalize=False,  # 是否对 embedding 做 L2 normalize。 #
        model_py_path=None,  # Stage2 模型结构文件路径，例如 model_stage2_mixed.py。 #
        model_kwargs=None,  # 构建 TahoeStage2MixedModel 的参数。 #
        num_bins=51,  # 表达量分桶数。 #
        contextual_value_mode="bin",  # contextual 表达值输入模式：bin / as_is / log1p。 #
        contextual_max_genes=None,  # contextual 模式送入 Stage2 encoder 的最大 gene 数；None 表示全部基因。 #
        contextual_gene_selection="matched_first",  # gene 选择策略：matched_first / first。 #
        contextual_fallback="static",  # 未送入 Stage2 的 gene embedding 填充方式：static / zero。 #
        **kwargs,  # 其他额外参数。 #
    ):  # 初始化函数结束。 #
        super().__init__(device=device, **kwargs)  # 调用父类初始化。 #
        self.ckpt_path = Path(ckpt_path)  # 保存 best.pt 路径。 #
        self.vocab_path = Path(vocab_path)  # 保存 vocab 路径。 #
        self.gears_hidden_size = int(gears_hidden_size)  # 保存 GEARS hidden_size。 #
        self.output_dim = int(gears_hidden_size)  # adapter 输出维度等于 GEARS hidden_size。 #
        self.raw_output_dim = None  # 保存 Stage2 原始 embedding 维度。 #
        self.missing_strategy = str(missing_strategy)  # 保存缺失基因填充策略。 #
        self.project_method = str(project_method)  # 保存投影方式。 #
        self.normalize = bool(normalize)  # 保存是否归一化。 #

        self.model_py_path = Path(model_py_path) if model_py_path is not None else None  # 保存 Stage2 模型结构文件路径。 #
        self.model_kwargs = model_kwargs or {}  # 保存模型构建参数。 #
        self.num_bins = int(num_bins)  # 保存表达量 bin 数。 #
        self.contextual_value_mode = str(contextual_value_mode)  # 保存 contextual 表达值输入模式。 #
        self.contextual_max_genes = None if contextual_max_genes is None else int(contextual_max_genes)  # 保存 contextual 最大 gene 数。 #
        self.contextual_gene_selection = str(contextual_gene_selection)  # 保存 contextual gene 选择策略。 #
        self.contextual_fallback = str(contextual_fallback)  # 保存未进入 Stage2 的 gene embedding 填充方式。 #
        self.stage2_model = None  # 初始化完整 Stage2 模型，contextual 模式才加载。 #

        self.symbol_to_id = {}  # 初始化 gene_symbol 到 token_id 的映射。 #
        self.symbol_upper_to_id = {}  # 初始化大写 gene_symbol 到 token_id 的映射。 #
        self.ensembl_to_id = {}  # 初始化 Ensembl 到 token_id 的映射。 #
        self.token_to_id = {}  # 初始化 token 到 token_id 的映射。 #
        self.gene_token_ids_all = []  # 初始化所有 gene token id。 #
        self.shared_embedding = None  # 初始化 shared_token_embedding.weight。 #
        self.gene_list = None  # 初始化 GEARS gene_list。 #
        self.pert_list = None  # 初始化 GEARS pert_list。 #
        self.gene_token_ids = None  # 初始化 gene token ids。 #
        self.pert_token_ids = None  # 初始化 pert token ids。 #

        self._load_vocab()  # 读取 vocab。 #
        self._load_shared_embedding()  # 读取 shared_token_embedding.weight。 #

    def _read_json_or_jsonl(self, path):  # 定义读取 json 或 jsonl 的函数。 #
        text = Path(path).read_text(encoding="utf-8")  # 读取文件文本。 #
        try:  # 尝试按完整 JSON 解析。 #
            obj = json.loads(text)  # 解析 JSON。 #
            if isinstance(obj, list):  # 如果 JSON 本身是列表。 #
                return obj  # 返回列表。 #
            if isinstance(obj, dict):  # 如果 JSON 是字典。 #
                return obj.get("genes", obj.get("items", obj.get("data", [])))  # 尝试取常见列表字段。 #
            return []  # 其他类型返回空列表。 #
        except Exception:  # 如果完整 JSON 解析失败。 #
            return [json.loads(line) for line in text.splitlines() if line.strip()]  # 按 jsonl 逐行解析。 #

    def _load_vocab(self):  # 定义 vocab 加载函数。 #
        if not self.vocab_path.exists():  # 检查 vocab 文件是否存在。 #
            raise FileNotFoundError(f"vocab_path not found: {self.vocab_path}")  # 不存在则报错。 #
        records = self._read_json_or_jsonl(self.vocab_path)  # 读取 vocab 记录。 #
        if not isinstance(records, list):  # 检查 records 是否为列表。 #
            raise ValueError("vocab_path must be a jsonl file or a json file containing a list.")  # 类型不对则报错。 #
        for item in records:  # 遍历 vocab 记录。 #
            if not isinstance(item, dict):  # 跳过非字典记录。 #
                continue  # 继续下一条。 #
            token_id = item.get("token_id", item.get("id", item.get("global_id", None)))  # 读取 token id。 #
            gene_symbol = item.get("gene_symbol", item.get("symbol", None))  # 读取 gene symbol。 #
            ensembl_id = item.get("ensembl_id", item.get("ensembl", None))  # 读取 Ensembl id。 #
            token = item.get("token", item.get("global_token", None))  # 读取 token 字段。 #
            if token_id is None:  # 如果没有 token id。 #
                continue  # 跳过该记录。 #
            token_id = int(token_id)  # 将 token id 转成 int。 #
            self.gene_token_ids_all.append(token_id)  # 保存 gene token id。 #
            if token is not None:  # 如果存在 token 字段。 #
                self.token_to_id[str(token)] = token_id  # 保存 token 到 id 映射。 #
            if gene_symbol is not None:  # 如果存在 gene symbol。 #
                gene_symbol = str(gene_symbol)  # 转成字符串。 #
                self.symbol_to_id[gene_symbol] = token_id  # 保存原始 symbol 映射。 #
                self.symbol_upper_to_id[gene_symbol.upper()] = token_id  # 保存大写 symbol 映射。 #
                self.token_to_id[f"<gene:{gene_symbol}>"] = token_id  # 保存 <gene:symbol> 映射。 #
                self.token_to_id[f"<gene:{gene_symbol.upper()}>"] = token_id  # 保存 <gene:SYMBOL> 映射。 #
            if ensembl_id is not None:  # 如果存在 Ensembl id。 #
                ensembl_id = str(ensembl_id)  # 转成字符串。 #
                self.ensembl_to_id[ensembl_id] = token_id  # 保存 Ensembl 映射。 #
                self.token_to_id[f"<gene:{ensembl_id}>"] = token_id  # 保存 <gene:ensembl> 映射。 #
        self.gene_token_ids_all = sorted(set(self.gene_token_ids_all))  # 去重并排序所有 gene token id。 #
        print(f"[Stage2Adapter] vocab records={len(records)}, gene_tokens={len(self.gene_token_ids_all)}, symbols={len(self.symbol_to_id)}, ensembl={len(self.ensembl_to_id)}")  # 打印 vocab 统计。 #

    def _extract_state_dict(self, ckpt):  # 定义 checkpoint 中提取 state_dict 的函数。 #
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:  # 如果 checkpoint 有 model_state_dict。 #
            return ckpt["model_state_dict"]  # 返回 model_state_dict。 #
        if isinstance(ckpt, dict) and "state_dict" in ckpt:  # 如果 checkpoint 有 state_dict。 #
            return ckpt["state_dict"]  # 返回 state_dict。 #
        if isinstance(ckpt, dict) and "model" in ckpt:  # 如果 checkpoint 有 model 字段。 #
            return ckpt["model"]  # 返回 model 字段。 #
        return ckpt  # 否则直接返回 ckpt。 #

    def _load_shared_embedding(self):  # 定义 shared embedding 加载函数。 #
        if not self.ckpt_path.exists():  # 检查 checkpoint 是否存在。 #
            raise FileNotFoundError(f"ckpt_path not found: {self.ckpt_path}")  # 不存在则报错。 #
        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")  # 读取 checkpoint 到 CPU。 #
        state = self._extract_state_dict(ckpt)  # 提取 state_dict。 #
        state = {str(k).replace("module.", ""): v for k, v in state.items()}  # 去掉 DataParallel 的 module 前缀。 #
        if "shared_token_embedding.weight" not in state:  # 检查是否存在 shared_token_embedding.weight。 #
            keys = [k for k in state.keys() if "embedding" in str(k).lower()]  # 收集可能的 embedding key。 #
            raise KeyError(f"Cannot find shared_token_embedding.weight. Embedding-like keys: {keys[:20]}")  # 报错并显示候选。 #
        emb = state["shared_token_embedding.weight"].detach().float().cpu()  # 提取 shared embedding。 #
        self.shared_embedding = emb  # 保存 shared embedding。 #
        self.raw_output_dim = int(emb.shape[1])  # 保存 Stage2 原始 embedding 维度。 #
        print(f"[Stage2Adapter] shared_token_embedding shape={tuple(emb.shape)}, adapter output_dim={self.output_dim}")  # 打印 embedding 形状。 #

    def setup(self, gene_list: List[str], pert_list: List[str], gene_id_list=None):  # 对齐 GEARS gene_list / gene_id_list / pert_list。 #
        self.gene_list = list(gene_list)  # 保存 GEARS gene symbol 列表。 #
        self.pert_list = list(pert_list)  # 保存 GEARS perturbation gene 列表。 #
        self.gene_id_list = list(gene_id_list) if gene_id_list is not None else None  # 保存 GEARS gene ID 列表。 #

        self.gene_token_ids = self._map_gene_tokens_by_best_coverage(  # 用覆盖率最高策略优先匹配 gene。 #
            gene_list=self.gene_list,  # 传入 gene symbol。 #
            gene_id_list=self.gene_id_list,  # 传入 gene ID。 #
        )  # gene token_id 映射结束。 #

        self.pert_token_ids = self._map_names_to_token_ids(self.pert_list)  # perturbation gene 仍然按 gene symbol 匹配。 #

        gene_matched = sum(x is not None for x in self.gene_token_ids)  # 统计 gene 最终匹配数。 #
        pert_matched = sum(x is not None for x in self.pert_token_ids)  # 统计 pert 匹配数。 #

        print(f"[Stage2Adapter] GEARS genes matched={gene_matched}/{len(self.gene_token_ids)}, missing={len(self.gene_token_ids) - gene_matched}")  # 打印 gene 匹配结果。 #
        print(f"[Stage2Adapter] GEARS perts matched={pert_matched}/{len(self.pert_token_ids)}, missing={len(self.pert_token_ids) - pert_matched}")  # 打印 pert 匹配结果。 #

        if gene_matched == 0:  # 如果 gene 一个都没有匹配上。 #
            raise ValueError("No GEARS gene matched Stage2 vocabulary. Please check gene_symbol / ensembl_id format.")  # 报错。 #

        if pert_matched == 0:  # 如果 pert 一个都没有匹配上。 #
            raise ValueError("No GEARS perturbation gene matched Stage2 vocabulary. Please check pert_list format.")  # 报错。 #

    def _lookup_token_id(self, name: str) -> Optional[int]:  # 根据基因名查找 token id。 #
        if name is None:  # 如果 name 为空。 #
            return None  # 返回 None。 #
        name = str(name)  # 转成字符串。 #
        if name in self.symbol_to_id:  # 如果直接匹配 gene symbol。 #
            return self.symbol_to_id[name]  # 返回 token id。 #
        if name.upper() in self.symbol_upper_to_id:  # 如果大写匹配 gene symbol。 #
            return self.symbol_upper_to_id[name.upper()]  # 返回 token id。 #
        if name in self.ensembl_to_id:  # 如果匹配 Ensembl id。 #
            return self.ensembl_to_id[name]  # 返回 token id。 #
        if name in self.token_to_id:  # 如果匹配 token 字段。 #
            return self.token_to_id[name]  # 返回 token id。 #
        if f"<gene:{name}>" in self.token_to_id:  # 如果匹配 <gene:name>。 #
            return self.token_to_id[f"<gene:{name}>"]  # 返回 token id。 #
        if f"<gene:{name.upper()}>" in self.token_to_id:  # 如果匹配 <gene:NAME>。 #
            return self.token_to_id[f"<gene:{name.upper()}>"]  # 返回 token id。 #
        return None  # 未匹配返回 None。 #

    def _map_names_to_token_ids(self, names: List[str]) -> List[Optional[int]]:  # 将基因名列表映射成 token id 列表。 #
        return [self._lookup_token_id(x) for x in names]  # 返回 token id 列表。 #


    def _strip_version(self, name):  # 去掉 Ensembl ID 或基因名末尾的版本号。 #
        if name is None:  # 如果输入为空。 #
            return None  # 返回 None。 #
        name = str(name)  # 转成字符串。 #
        if "." in name:  # 如果包含点号。 #
            left, right = name.rsplit(".", 1)  # 从右边切分一次。 #
            if right.isdigit():  # 如果点号后面是数字。 #
                return left  # 返回去掉版本号后的 ID。 #
        return name  # 否则返回原始名称。 #

    def _lookup_gene_token_by_strategy(self, gene_name, gene_id, strategy):  # 按指定策略查 token_id。 #
        gene_name = None if gene_name is None else str(gene_name)  # 规范化 gene_name。 #
        gene_id = None if gene_id is None else str(gene_id)  # 规范化 gene_id。 #

        if strategy == "var_index_ensembl":  # 如果策略是直接用 var.index 的 Ensembl ID。 #
            if gene_id in self.ensembl_to_id:  # 如果 Ensembl ID 在 vocab 中。 #
                return self.ensembl_to_id[gene_id]  # 返回 token_id。 #
            if f"<gene:{gene_id}>" in self.token_to_id:  # 如果 token 形式在 vocab 中。 #
                return self.token_to_id[f"<gene:{gene_id}>"]  # 返回 token_id。 #
            return None  # 否则返回 None。 #

        if strategy == "var_index_ensembl_strip":  # 如果策略是去版本号后的 Ensembl ID。 #
            gene_id_strip = self._strip_version(gene_id)  # 去掉版本号。 #
            if gene_id_strip in self.ensembl_to_id:  # 如果去版本号 ID 在 vocab 中。 #
                return self.ensembl_to_id[gene_id_strip]  # 返回 token_id。 #
            if f"<gene:{gene_id_strip}>" in self.token_to_id:  # 如果 token 形式在 vocab 中。 #
                return self.token_to_id[f"<gene:{gene_id_strip}>"]  # 返回 token_id。 #
            return None  # 否则返回 None。 #

        if strategy == "gene_symbol_exact":  # 如果策略是 gene symbol 精确匹配。 #
            if gene_name in self.symbol_to_id:  # 如果 symbol 在 vocab 中。 #
                return self.symbol_to_id[gene_name]  # 返回 token_id。 #
            if f"<gene:{gene_name}>" in self.token_to_id:  # 如果 token 形式在 vocab 中。 #
                return self.token_to_id[f"<gene:{gene_name}>"]  # 返回 token_id。 #
            return None  # 否则返回 None。 #

        if strategy == "gene_symbol_upper":  # 如果策略是 gene symbol 大写匹配。 #
            gene_name_upper = None if gene_name is None else gene_name.upper()  # 转成大写。 #
            if gene_name_upper in self.symbol_upper_to_id:  # 如果大写 symbol 在 vocab 中。 #
                return self.symbol_upper_to_id[gene_name_upper]  # 返回 token_id。 #
            if f"<gene:{gene_name_upper}>" in self.token_to_id:  # 如果 token 形式在 vocab 中。 #
                return self.token_to_id[f"<gene:{gene_name_upper}>"]  # 返回 token_id。 #
            return None  # 否则返回 None。 #

        raise ValueError(f"Unknown gene match strategy: {strategy}")  # 未知策略则报错。 #

    def _map_gene_tokens_by_best_coverage(self, gene_list, gene_id_list=None):  # 先按覆盖率选择主策略，再用其他策略补充。 #
        gene_list = list(gene_list)  # 转成 list。 #
        gene_id_list = list(gene_id_list) if gene_id_list is not None else [None] * len(gene_list)  # 如果没有 gene_id，则用 None 占位。 #

        if len(gene_list) != len(gene_id_list):  # 检查长度是否一致。 #
            raise ValueError(f"gene_list and gene_id_list length mismatch: {len(gene_list)} vs {len(gene_id_list)}")  # 长度不一致则报错。 #

        strategies = [  # 定义候选匹配策略。 #
            "var_index_ensembl",  # 使用 var.index 原始 Ensembl ID。 #
            "var_index_ensembl_strip",  # 使用去版本号 Ensembl ID。 #
            "gene_symbol_exact",  # 使用 gene symbol 精确匹配。 #
            "gene_symbol_upper",  # 使用 gene symbol 大写匹配。 #
        ]  # 策略列表结束。 #

        if all(x is None for x in gene_id_list):  # 如果没有 gene_id_list。 #
            strategies = ["gene_symbol_exact", "gene_symbol_upper"]  # 只使用 symbol 相关策略。 #

        strategy_to_token_ids = {}  # 保存每个策略的 token_id 结果。 #
        strategy_summary = []  # 保存每个策略的覆盖率。 #

        for strategy in strategies:  # 遍历每个策略。 #
            token_ids = [  # 逐个 gene 匹配。 #
                self._lookup_gene_token_by_strategy(gene_name, gene_id, strategy)  # 查找 token_id。 #
                for gene_name, gene_id in zip(gene_list, gene_id_list)  # 同时遍历 gene_name 和 gene_id。 #
            ]  # token_id 列表结束。 #
            matched = sum(x is not None for x in token_ids)  # 统计匹配数量。 #
            strategy_to_token_ids[strategy] = token_ids  # 保存该策略结果。 #
            strategy_summary.append({"strategy": strategy, "matched": matched, "total": len(gene_list), "rate": matched / len(gene_list)})  # 保存覆盖率。 #

        strategy_summary = sorted(strategy_summary, key=lambda x: x["matched"], reverse=True)  # 按覆盖率从高到低排序。 #
        ordered_strategies = [x["strategy"] for x in strategy_summary]  # 得到最终使用顺序。 #

        final_token_ids = [None] * len(gene_list)  # 初始化最终 token_id。 #
        final_sources = ["missing"] * len(gene_list)  # 初始化每个 gene 的来源策略。 #

        for strategy in ordered_strategies:  # 按覆盖率从高到低依次补充。 #
            token_ids = strategy_to_token_ids[strategy]  # 取出该策略结果。 #
            for i, token_id in enumerate(token_ids):  # 遍历每个 gene。 #
                if final_token_ids[i] is None and token_id is not None:  # 只补充仍未匹配的位置。 #
                    final_token_ids[i] = token_id  # 写入 token_id。 #
                    final_sources[i] = strategy  # 记录来源策略。 #

        final_matched = sum(x is not None for x in final_token_ids)  # 统计最终匹配数量。 #
        self.gene_match_summary = strategy_summary  # 保存策略覆盖率。 #
        self.gene_match_sources = final_sources  # 保存每个 gene 的匹配来源。 #

        print("[Stage2Adapter] gene matching strategy coverage:")  # 打印策略覆盖率标题。 #
        for item in strategy_summary:  # 遍历策略统计。 #
            print(f"  {item['strategy']}: {item['matched']}/{item['total']} = {item['rate']:.4f}")  # 打印每个策略覆盖率。 #
        print(f"[Stage2Adapter] selected primary gene strategy: {ordered_strategies[0]}")  # 打印主策略。 #
        print(f"[Stage2Adapter] final gene matched after supplement: {final_matched}/{len(gene_list)} = {final_matched / len(gene_list):.4f}")  # 打印最终匹配率。 #

        return final_token_ids  # 返回最终 token_id。 #
    

    def _missing_fill_vector(self) -> torch.Tensor:  # 构造缺失基因填充向量。 #
        if self.missing_strategy == "zero":  # 如果缺失策略是 zero。 #
            return torch.zeros(self.raw_output_dim)  # 返回零向量。 #
        if self.missing_strategy == "mean_all":  # 如果缺失策略是 mean_all。 #
            return self.shared_embedding.mean(dim=0)  # 返回所有 token embedding 均值。 #
        valid_ids = [i for i in self.gene_token_ids_all if 0 <= int(i) < self.shared_embedding.shape[0]]  # 保留合法 gene token id。 #
        if len(valid_ids) == 0:  # 如果没有合法 gene token。 #
            return self.shared_embedding.mean(dim=0)  # 回退到所有 token 均值。 #
        return self.shared_embedding[torch.tensor(valid_ids, dtype=torch.long)].mean(dim=0)  # 返回 gene token embedding 均值。 #

    def _embedding_by_token_ids(self, token_ids: List[Optional[int]]) -> torch.Tensor:  # 根据 token ids 提取 embedding。 #
        fill = self._missing_fill_vector()  # 获取缺失填充向量。 #
        max_id = self.shared_embedding.shape[0] - 1  # 获取最大合法 token id。 #
        out = []  # 初始化输出列表。 #
        for tid in token_ids:  # 遍历 token id。 #
            if tid is None or int(tid) < 0 or int(tid) > max_id:  # 如果缺失或越界。 #
                out.append(fill)  # 使用填充值。 #
            else:  # 如果 token id 合法。 #
                out.append(self.shared_embedding[int(tid)])  # 取对应 embedding。 #
        return torch.stack(out, dim=0)  # 返回 [N, raw_dim]。 #

    def _project_to_hidden(self, emb: torch.Tensor) -> torch.Tensor:  # 在 adapter 内将 raw_dim 投影到 GEARS hidden_size。 #
        if emb.shape[1] == self.gears_hidden_size:  # 如果已经是 hidden_size。 #
            out = emb  # 直接使用。 #
        elif self.project_method == "slice":  # 如果使用截断/补零投影。 #
            if emb.shape[1] > self.gears_hidden_size:  # 如果原始维度大于 hidden_size。 #
                out = emb[:, : self.gears_hidden_size].contiguous()  # 截取前 hidden_size 维。 #
            else:  # 如果原始维度小于 hidden_size。 #
                pad = torch.zeros(emb.shape[0], self.gears_hidden_size - emb.shape[1], dtype=emb.dtype, device=emb.device)  # 构造补零矩阵。 #
                out = torch.cat([emb, pad], dim=1)  # 拼接补零矩阵。 #
        elif self.project_method == "mean_pool":  # 如果使用分块均值池化。 #
            if emb.shape[1] % self.gears_hidden_size != 0:  # 如果原始维度不能整除 hidden_size。 #
                raise ValueError("mean_pool requires raw_output_dim % gears_hidden_size == 0.")  # 抛出错误。 #
            group = emb.shape[1] // self.gears_hidden_size  # 计算每组维度数。 #
            out = emb.reshape(emb.shape[0], self.gears_hidden_size, group).mean(dim=2)  # 分块均值池化。 #
        else:  # 如果投影方式未知。 #
            raise ValueError(f"Unknown project_method: {self.project_method}")  # 抛出错误。 #

        if self.normalize:  # 如果需要 L2 normalize。 #
            out = torch.nn.functional.normalize(out, p=2, dim=1)  # 对 embedding 做 L2 normalize。 #

        return out  # 返回 [N, hidden_size]。 #

    def get_static_gene_embeddings(self, gene_list: List[str]) -> torch.Tensor:  # 返回 static gene embedding。 #
        if self.gene_token_ids is None:  # 检查是否已经 setup。 #
            raise RuntimeError("Please call setup(gene_list, pert_list, gene_id_list) before get_static_gene_embeddings().")  # 未 setup 则报错。 #

        if len(gene_list) == len(self.gene_token_ids):  # 如果输入 gene 数和 setup 时一致。 #
            token_ids = self.gene_token_ids  # 使用覆盖率优先策略得到的 token_id。 #
        else:  # 如果长度不一致。 #
            token_ids = self._map_names_to_token_ids(list(gene_list))  # 回退到 symbol 匹配。 #

        emb = self._embedding_by_token_ids(token_ids)  # 根据 token_id 提取原始 embedding。 #
        emb = self._project_to_hidden(emb)  # 在 adapter 内投影到 GEARS hidden_size。 #
        return emb.clone()  # 返回 embedding 副本。 #

    def get_pert_embeddings(self, pert_list: List[str]) -> torch.Tensor:  # 返回 perturbation embedding。 #
        if self.pert_list is None:  # 检查是否已经 setup。 #
            raise RuntimeError("Please call setup(gene_list, pert_list, gene_id_list) before get_pert_embeddings().")  # 未 setup 则报错。 #

        token_ids = self._map_names_to_token_ids(list(pert_list))  # 根据 pert_list 映射 token ids。 #
        emb = self._embedding_by_token_ids(token_ids)  # 根据 token_id 提取原始 embedding。 #
        emb = self._project_to_hidden(emb)  # 在 adapter 内投影到 GEARS hidden_size。 #
        return emb.clone()  # 返回 embedding 副本。 #


    def _load_stage2_model_class(self):  # 从 model_py_path 动态加载 Stage2 模型类。 #
        if self.model_py_path is None:  # 如果没有传入模型结构文件。 #
            raise ValueError("scfm_contextual requires model_py_path.")  # 抛出错误。 #
        if not self.model_py_path.exists():  # 如果模型结构文件不存在。 #
            raise FileNotFoundError(f"model_py_path not found: {self.model_py_path}")  # 抛出错误。 #

        model_dir = str(self.model_py_path.parent)  # 获取模型文件所在目录。 #
        if model_dir not in sys.path:  # 如果模型目录不在 sys.path。 #
            sys.path.insert(0, model_dir)  # 加入 sys.path，避免模型文件内部 import 失败。 #

        spec = importlib.util.spec_from_file_location("stage2_model_module", str(self.model_py_path))  # 构建动态导入 spec。 #
        module = importlib.util.module_from_spec(spec)  # 构建模块对象。 #
        spec.loader.exec_module(module)  # 执行模块导入。 #

        class_name = getattr(self, "model_class_name", "TahoeStage2MixedModel")  # 获取模型类名。 #
        if not hasattr(module, class_name):  # 如果模型文件里没有这个类。 #
            available = [x for x in dir(module) if "Model" in x or "model" in x]  # 收集可能的模型类名。 #
            raise AttributeError(f"Cannot find class {class_name} in {self.model_py_path}. Available model-like names: {available}")  # 抛出错误。 #

        return getattr(module, class_name)  # 返回模型类。 #

    def _ensure_stage2_model_loaded(self):  # 确保完整 Stage2 模型已经加载。 #
        if self.stage2_model is not None:  # 如果已经加载。 #
            return  # 直接返回。 #

        model_cls = self._load_stage2_model_class()  # 加载 Stage2 模型类。 #
        model_kwargs = dict(getattr(self, "model_kwargs", {}) or {})  # 复制模型初始化参数。 #

        model_kwargs.setdefault("global_vocab_size", int(self.shared_embedding.shape[0]))  # 默认设置词表大小。 #
        model_kwargs.setdefault("d_model", int(self.raw_output_dim))  # 默认设置 d_model。 #

        print(f"[Stage2Adapter] loading contextual Stage2 model with kwargs={model_kwargs}")  # 打印模型参数。 #
        model = model_cls(**model_kwargs)  # 初始化 Stage2 模型。 #

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")  # 读取 checkpoint。 #
        state = self._extract_state_dict(ckpt)  # 提取 state_dict。 #
        state = {str(k).replace("module.", ""): v for k, v in state.items()}  # 去掉 module. 前缀。 #

        model_state = model.state_dict()  # 获取当前模型的 state_dict。 #
        compatible_state = {}  # 保存 key 存在且 shape 完全匹配的 checkpoint 参数。 #
        skipped_shape = []  # 保存 shape 不匹配而跳过的参数。 #
        ckpt_unexpected_keys = []  # 保存 checkpoint 中存在、但当前模型中不存在的参数。 #

        for k, v in state.items():  # 遍历 checkpoint 参数。 #
            if k not in model_state:  # 如果当前模型没有这个 key。 #
                ckpt_unexpected_keys.append(k)  # 记录 checkpoint 多余参数。 #
                continue  # 跳过。 #
            if hasattr(v, "shape") and tuple(v.shape) != tuple(model_state[k].shape):  # 如果 shape 不一致。 #
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))  # 记录 shape mismatch。 #
                continue  # 跳过。 #
            compatible_state[k] = v  # key 和 shape 都匹配则保留。 #

        # ------------------------------------------------------------
        # Encoder-side 定义：
        # contextual embedding 真正经过以下模块，因此这些参数必须 100% 正确加载。
        # Decoder / LM head 等其他参数允许因为算法变化而 missing / unexpected / shape mismatch。
        # ------------------------------------------------------------
        critical_encoder_prefixes = (
            "shared_token_embedding.",
            "token_norm.",
            "value_encoder.",
            "mask_flag_embedding.",
            "encoder.",
        )

        def is_critical_encoder_key(key: str) -> bool:  # 判断参数是否属于 contextual encoder 路径。 #
            return str(key).startswith(critical_encoder_prefixes)  # 返回判断结果。 #

        # 1. 在真正 load_state_dict 之前先检查 Encoder shape mismatch。
        bad_encoder_shape = [
            item for item in skipped_shape
            if is_critical_encoder_key(item[0])
        ]

        if bad_encoder_shape:
            details = "\n".join(
                f"  {k}: ckpt={ckpt_shape}, model={model_shape}"
                for k, ckpt_shape, model_shape in bad_encoder_shape[:100]
            )
            raise RuntimeError(
                "[Stage2Adapter] Critical Encoder parameters have shape mismatch. "
                "Contextual model loading is aborted.\n"
                f"{details}"
            )

        # 2. 当前 model.py 中所有 Encoder-side 参数都必须存在于 checkpoint，
        #    且必须进入 compatible_state；否则禁止继续。
        expected_encoder_keys = [
            k for k in model_state.keys()
            if is_critical_encoder_key(k)
        ]

        not_loaded_encoder_keys = [
            k for k in expected_encoder_keys
            if k not in compatible_state
        ]

        if not_loaded_encoder_keys:
            details = "\n".join(f"  {k}" for k in not_loaded_encoder_keys[:200])
            raise RuntimeError(
                "[Stage2Adapter] Some critical Encoder parameters are missing "
                "from the checkpoint or were not compatible. "
                "Contextual model loading is aborted.\n"
                f"{details}"
            )

        # 3. 加载全部兼容参数。
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)  # Decoder 允许非严格加载。 #

        print("[Stage2Adapter] contextual model loaded with safe filtering.")  # 打印加载方式。 #
        print(
            f"[Stage2Adapter] loaded keys={len(compatible_state)}, "
            f"missing={len(missing)}, "
            f"unexpected={len(unexpected)}, "
            f"skipped_shape={len(skipped_shape)}, "
            f"ckpt_unexpected={len(ckpt_unexpected_keys)}"
        )  # 打印统计。 #

        if len(skipped_shape) > 0:  # 如果有 shape 不匹配的参数。 #
            print("[Stage2Adapter] first shape-mismatched skipped keys:")  # 打印标题。 #
            for item in skipped_shape[:20]:  # 打印前 20 个。 #
                print(f"  {item[0]}: ckpt={item[1]}, model={item[2]}")  # 打印 key 和 shape。 #

        # 4. load_state_dict 返回的 missing 中不能包含任何 Encoder-side 参数。
        missing_encoder_keys = [
            k for k in missing
            if is_critical_encoder_key(k)
        ]

        if missing_encoder_keys:
            details = "\n".join(f"  {k}" for k in missing_encoder_keys[:200])
            raise RuntimeError(
                "[Stage2Adapter] Critical Encoder parameters were reported as missing "
                "after load_state_dict. Contextual model loading is aborted.\n"
                f"{details}"
            )

        # 5. 对 Encoder 参数做逐 tensor 精确验证。
        #    这里必须在 model.half() 之前执行，否则 fp16 转换会导致与 fp32 checkpoint 不再逐元素完全相等。
        loaded_model_state = model.state_dict()
        verification_failed = []

        for key in expected_encoder_keys:
            ckpt_tensor = compatible_state[key]
            model_tensor = loaded_model_state[key]

            if not torch.is_tensor(ckpt_tensor) or not torch.is_tensor(model_tensor):
                verification_failed.append((key, "non-tensor state value"))
                continue

            ckpt_tensor_cpu = ckpt_tensor.detach().cpu()
            model_tensor_cpu = model_tensor.detach().cpu()

            if (
                ckpt_tensor_cpu.shape != model_tensor_cpu.shape
                or ckpt_tensor_cpu.dtype != model_tensor_cpu.dtype
                or not torch.equal(ckpt_tensor_cpu, model_tensor_cpu)
            ):
                verification_failed.append(
                    (
                        key,
                        f"ckpt_shape={tuple(ckpt_tensor_cpu.shape)}, "
                        f"model_shape={tuple(model_tensor_cpu.shape)}, "
                        f"ckpt_dtype={ckpt_tensor_cpu.dtype}, "
                        f"model_dtype={model_tensor_cpu.dtype}",
                    )
                )

        if verification_failed:
            details = "\n".join(
                f"  {key}: {reason}"
                for key, reason in verification_failed[:100]
            )
            raise RuntimeError(
                "[Stage2Adapter] Exact tensor verification failed for critical "
                "Encoder parameters. Contextual model loading is aborted.\n"
                f"{details}"
            )

        loaded_encoder_keys = [
            k for k in compatible_state.keys()
            if is_critical_encoder_key(k)
        ]

        print(
            "[Stage2Adapter] Encoder loading check passed: "
            f"{len(loaded_encoder_keys)}/{len(expected_encoder_keys)} "
            "critical Encoder parameters loaded."
        )
        print(
            "[Stage2Adapter] Exact tensor verification passed for all "
            "critical Encoder parameters."
        )

        # 6. Decoder-side 的 missing / unexpected / shape mismatch 只打印，不阻断。
        decoder_or_other_missing = [
            k for k in missing
            if not is_critical_encoder_key(k)
        ]
        decoder_or_other_shape_skips = [
            item for item in skipped_shape
            if not is_critical_encoder_key(item[0])
        ]

        if decoder_or_other_missing:
            print(
                "[Stage2Adapter] non-Encoder missing keys are allowed "
                f"for decoder algorithm changes: {len(decoder_or_other_missing)}"
            )
            for k in decoder_or_other_missing[:20]:
                print(f"  missing non-Encoder key: {k}")

        if decoder_or_other_shape_skips:
            print(
                "[Stage2Adapter] non-Encoder shape mismatches are allowed "
                f"for decoder algorithm changes: {len(decoder_or_other_shape_skips)}"
            )

        if ckpt_unexpected_keys:
            print(
                "[Stage2Adapter] checkpoint-only keys are allowed "
                f"for decoder algorithm changes: {len(ckpt_unexpected_keys)}"
            )
            for k in ckpt_unexpected_keys[:20]:
                print(f"  checkpoint-only key: {k}")

        model.to(self.device)  # 移动到 GPU。 #

        if str(self.device).startswith("cuda"):  # 如果使用 GPU。 #
            model.half()  # Stage2 模型转 fp16，减少显存。 #

        model.eval()  # eval 模式。 #

        for p in model.parameters():  # 遍历 Stage2 参数。 #
            p.requires_grad = False  # 冻结 Stage2 参数，不参与 GEARS 训练。 #

        self.stage2_model = model  # 保存 Stage2 模型。 #
    def _select_contextual_gene_indices(self, token_ids: List[Optional[int]], n_genes: int) -> torch.Tensor:  # 定义函数：选择送入 Stage2 encoder 的 gene 下标。 #
        if self.contextual_max_genes is None:  # 如果没有设置最大 gene 数。 #
            return torch.arange(n_genes, dtype=torch.long)  # 返回全部 gene 下标。 #
        if self.contextual_max_genes <= 0:  # 如果最大 gene 数小于等于 0。 #
            return torch.arange(n_genes, dtype=torch.long)  # 返回全部 gene 下标。 #
        if self.contextual_max_genes >= n_genes:  # 如果最大 gene 数大于等于总 gene 数。 #
            return torch.arange(n_genes, dtype=torch.long)  # 返回全部 gene 下标。 #

        mode = self.contextual_gene_selection.lower()  # 获取 gene 选择策略。 #
        max_genes = int(self.contextual_max_genes)  # 获取最大 gene 数。 #

        if mode == "first":  # 如果选择当前 GEARS gene_list 的前 max_genes 个基因。 #
            indices = list(range(max_genes))  # 构造前 max_genes 个下标。 #
            return torch.tensor(indices, dtype=torch.long)  # 返回 tensor。 #

        if mode == "matched_first":  # 如果优先选择能匹配 Stage2 vocab 的 gene。 #
            matched = [i for i, tid in enumerate(token_ids) if tid is not None]  # 获取匹配到 token_id 的 gene 下标。 #
            missing = [i for i, tid in enumerate(token_ids) if tid is None]  # 获取没有匹配到 token_id 的 gene 下标。 #
            indices = (matched + missing)[:max_genes]  # 优先保留 matched gene，不足再用 missing gene 补齐。 #
            return torch.tensor(indices, dtype=torch.long)  # 返回 tensor。 #

        raise ValueError(f"Unknown contextual_gene_selection: {self.contextual_gene_selection}")  # 未知策略直接报错。 #

    def _build_contextual_inputs(self, x: torch.Tensor, gene_list: List[str]):  # 构建 Stage2 contextual 输入。 #
        device = self.device  # 获取设备。 #
        x = x.to(device).float()  # 表达矩阵移动到设备并转 float。 #
        n_genes = len(gene_list)  # 获取 GEARS 当前 gene 数。 #

        if self.gene_token_ids is not None and len(gene_list) == len(self.gene_token_ids):  # 如果 setup 中已有完整 gene_token_ids。 #
            token_ids = self.gene_token_ids  # 使用覆盖率优先策略得到的 token_id。 #
        else:  # 如果没有可用的 gene_token_ids。 #
            token_ids = self._map_names_to_token_ids(list(gene_list))  # 回退到 symbol 匹配。 #

        selected_idx_cpu = self._select_contextual_gene_indices(token_ids, n_genes)  # 选择送入 Stage2 的 gene 下标，仍在 CPU。 #
        selected_idx = selected_idx_cpu.to(device)  # 将 gene 下标移动到当前设备。 #

        x_selected = x.index_select(dim=1, index=selected_idx)  # 只取被选中的 gene 表达矩阵，shape 为 [B, K]。 #
        token_ids_selected = [token_ids[i] for i in selected_idx_cpu.tolist()]  # 取被选中 gene 对应的 token_id。 #

        pad_token_id = int(getattr(self.stage2_model, "pad_token_id", 0))  # 获取 pad token id。 #
        input_ids = [int(tid) if tid is not None else pad_token_id for tid in token_ids_selected]  # 缺失 gene 用 pad token。 #
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)  # 转成 tensor。 #
        input_ids = input_ids.unsqueeze(0).expand(x.shape[0], -1).contiguous()  # 扩展到 batch 维度，shape 为 [B, K]。 #

        missing_mask = torch.tensor([tid is None for tid in token_ids_selected], dtype=torch.bool, device=device)  # 构建缺失 gene mask。 #
        key_padding_mask = missing_mask.unsqueeze(0).expand(x.shape[0], -1).contiguous()  # 扩展到 batch 维度，shape 为 [B, K]。 #

        encoder_values = self._encode_contextual_values(x_selected)  # 根据 contextual_value_mode 处理表达值，shape 为 [B, K]。 #

        return input_ids, encoder_values, key_padding_mask, selected_idx, token_ids  # 返回 Stage2 输入、被选 gene 下标和完整 token_ids。 #

    def _build_contextual_fallback_embeddings(self, token_ids: List[Optional[int]], batch_size: int) -> torch.Tensor:  # 定义函数：构建完整 gene 数的 fallback embedding。 #
        device = self.device  # 获取设备。 #
        n_genes = len(token_ids)  # 获取完整 gene 数。 #

        if self.contextual_fallback.lower() == "zero":  # 如果 fallback 使用零向量。 #
            full_emb = torch.zeros(n_genes, self.gears_hidden_size, dtype=torch.float32, device=device)  # 构造 [N, H] 零向量。 #
        elif self.contextual_fallback.lower() == "static":  # 如果 fallback 使用 static scFM embedding。 #
            raw_emb = self._embedding_by_token_ids(token_ids).to(device).float()  # 根据完整 token_ids 取 raw static embedding。 #
            full_emb = self._project_to_hidden(raw_emb).to(device).float()  # 投影到 GEARS hidden_size。 #
        else:  # 如果 fallback 参数未知。 #
            raise ValueError(f"Unknown contextual_fallback: {self.contextual_fallback}")  # 直接报错。 #

        full_emb = full_emb.unsqueeze(0).expand(batch_size, -1, -1).contiguous().clone()  # 扩展到 [B, N, H]。 #
        return full_emb  # 返回完整 fallback embedding。 #
        
    def _encode_contextual_values(self, x: torch.Tensor) -> torch.Tensor:  # 定义函数：处理 contextual 输入表达值。 #
        mode = self.contextual_value_mode.lower()  # 转成小写，避免大小写问题。 #
        if mode == "bin":  # 如果使用原始 bin 模式。 #
            return left_binning(x, self.num_bins).float()  # 使用原来的分位数 bin。 #
        if mode in ["as_is", "none", "raw"]:  # 如果关闭 bin，直接使用当前表达值。 #
            return x.float()  # 直接返回 adata.X / cell_graphs.pkl 中的表达值。 #
        if mode == "log1p":  # 如果希望在 adapter 内做 log1p。 #
            return torch.log1p(torch.clamp(x.float(), min=0.0))  # 对非负表达值做 log1p。 #
        raise ValueError(f"Unknown contextual_value_mode: {self.contextual_value_mode}")  # 未知模式直接报错。 #

    def get_contextual_gene_embeddings(self, x: torch.Tensor, gene_list: List[str]):  # 返回 contextual gene embedding。 #
        self._ensure_stage2_model_loaded()  # 确保完整 Stage2 模型已加载。 #
        input_ids, encoder_values, key_padding_mask, selected_idx, token_ids = self._build_contextual_inputs(x, gene_list)  # 构建 encoder 输入。 #

        with torch.no_grad():  # 不对 Stage2 反向传播，节省显存。 #
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):  # Stage2 encoder 使用 fp16。 #
                encoder_outputs, _ = self.stage2_model.encode(  # 调用 Stage2 encoder。 #
                    encoder_input_gene_ids=input_ids,  # gene token ids，shape 为 [B, K]。 #
                    encoder_input_values=encoder_values,  # 表达值，shape 为 [B, K]。 #
                    encoder_key_padding_mask=key_padding_mask,  # padding mask，shape 为 [B, K]。 #
                    return_last_attn=False,  # 不返回 attention。 #
                )  # encoder 输出，shape 为 [B, K, raw_dim]。 #

        encoder_outputs = encoder_outputs.detach().float().clone()  # 转成 float32，供 GEARS 后续使用。 #
        batch_size = encoder_outputs.shape[0]  # 获取 batch size。 #
        selected_gene_num = encoder_outputs.shape[1]  # 获取 Stage2 实际处理的 gene 数 K。 #
        raw_dim = encoder_outputs.shape[2]  # 获取 Stage2 原始输出维度。 #

        contextual_emb = encoder_outputs.reshape(batch_size * selected_gene_num, raw_dim)  # 展平成 [B*K, raw_dim]。 #
        contextual_emb = self._project_to_hidden(contextual_emb)  # 投影到 [B*K, hidden_size]。 #
        contextual_emb = contextual_emb.reshape(batch_size, selected_gene_num, self.gears_hidden_size)  # 恢复成 [B, K, hidden_size]。 #

        if selected_gene_num == len(gene_list):  # 如果没有做 gene 长度限制。 #
            return contextual_emb  # 直接返回完整 contextual embedding。 #

        full_emb = self._build_contextual_fallback_embeddings(token_ids, batch_size=batch_size)  # 构建完整 [B, N, H] fallback embedding。 #
        full_emb[:, selected_idx, :] = contextual_emb  # 把 Stage2 contextual 输出写回对应 gene 位置。 #

        return full_emb  # 返回完整 [B, num_genes, hidden_size] embedding。 #