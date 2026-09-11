import torch
import torch.nn as nn
from torch_geometric.nn import SGConv


class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1]) if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU(),
            ]
        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GEARS_Model(torch.nn.Module):
    def __init__(self, args, adapter=None):
        super(GEARS_Model, self).__init__()

        self.args = args
        self.adapter = adapter

        self.num_genes = args["num_genes"]
        self.num_perts = args["num_perts"]
        hidden_size = args["hidden_size"]
        self.hidden_size = hidden_size

        self.gene_embedding_mode = args.get("gene_embedding_mode", args.get("gene_embedding_source", "native"))
        self.pert_embedding_mode = args.get("pert_embedding_mode", args.get("pert_embedding_source", "native"))
        self.gene_list = args.get("gene_list", None)
        self.pert_list = args.get("pert_list", None)
        self.scfm_dim = args.get("scfm_dim", None)

        self.uncertainty = args["uncertainty"]
        self.num_layers = args["num_go_gnn_layers"]
        self.indv_out_hidden_size = args["decoder_hidden_size"]
        self.num_layers_gene_pos = args["num_gene_gnn_layers"]
        self.no_perturb = args["no_perturb"]
        self.pert_emb_lambda = 0.2

        self.pert_w = nn.Linear(1, hidden_size)

        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        self.scfm_gene_projector = None

        if self.gene_embedding_mode in ["scfm_static", "scfm_contextual"]:  # 如果使用外部基模 gene embedding。 #
            if self.adapter is None:  # 检查 adapter 是否传入。 #
                raise ValueError(f"gene_embedding_mode='{self.gene_embedding_mode}' requires an adapter.")  # 没有 adapter 就报错。 #
            if self.scfm_dim is None:  # 检查 adapter 输出维度是否存在。 #
                raise ValueError(f"scfm_dim must be provided when using {self.gene_embedding_mode}.")  # 没有 scfm_dim 就报错。 #
            if self.scfm_dim != hidden_size:  # 如果 adapter 输出维度不是 GEARS hidden_size。 #
                raise ValueError(  # 抛出维度错误。 #
                    f"Adapter output_dim/scfm_dim must equal GEARS hidden_size when projector is inside adapter. "  # 提示 projector 在 adapter 内。 #
                    f"Got scfm_dim={self.scfm_dim}, hidden_size={hidden_size}. "  # 显示实际维度。 #
                    "Please project embeddings to hidden_size inside the adapter."  # 要求在 adapter 内投影。 #
                )  # 错误信息结束。 #
        
            self.scfm_gene_projector = nn.Identity()  # adapter 已经完成投影，GEARS 内部不再做 Linear。 #
        
            if self.gene_embedding_mode == "scfm_static":  # 如果使用静态 gene embedding。 #
                raw_gene_emb = self.adapter.get_static_gene_embeddings(self.gene_list)  # 从 adapter 获取已对齐、已投影的 gene embedding。 #
                if raw_gene_emb is None:  # 检查 adapter 是否返回 None。 #
                    raise ValueError("Adapter returned None for static gene embeddings.")  # 返回 None 就报错。 #
                if not torch.is_tensor(raw_gene_emb):  # 如果返回的不是 torch.Tensor。 #
                    raw_gene_emb = torch.tensor(raw_gene_emb)  # 转成 torch.Tensor。 #
                if raw_gene_emb.shape[0] != self.num_genes:  # 检查 gene 数量是否等于 GEARS num_genes。 #
                    raise ValueError(  # 抛出数量错误。 #
                        f"raw_gene_emb first dim should be num_genes={self.num_genes}, but got {raw_gene_emb.shape[0]}"  # 显示实际数量。 #
                    )  # 错误信息结束。 #
                if raw_gene_emb.shape[1] != hidden_size:  # 检查 embedding 维度是否等于 hidden_size。 #
                    raise ValueError(  # 抛出维度错误。 #
                        f"raw_gene_emb second dim should be hidden_size={hidden_size}, but got {raw_gene_emb.shape[1]}"  # 显示实际维度。 #
                    )  # 错误信息结束。 #
                self.register_buffer("raw_static_scfm_gene_emb", raw_gene_emb.detach().float())  # 注册静态 gene embedding buffer。 #

        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act="ReLU")
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act="ReLU")

        self.G_coexpress = args["G_coexpress"].to(args["device"])
        self.G_coexpress_weight = args["G_coexpress_weight"].to(args["device"])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for _ in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))

        self.G_sim = args["G_go"].to(args["device"])
        self.G_sim_weight = args["G_go_weight"].to(args["device"])

        self.sim_layers = torch.nn.ModuleList()
        for _ in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        self.recovery_w = MLP([hidden_size, hidden_size * 2, hidden_size], last_layer_act="linear")

        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes, hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        self.cross_gene_state = MLP([self.num_genes, hidden_size, hidden_size])

        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes, hidden_size + 1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size * 2, hidden_size, 1], last_layer_act="linear")

    def get_gene_embedding_for_batch(self, data, num_graphs):
        device = self.args["device"]

        if self.gene_embedding_mode == "native":
            gene_ids = torch.arange(self.num_genes, device=device).repeat(num_graphs)
            return self.gene_emb(gene_ids)

        if self.gene_embedding_mode == "scfm_static":
            raw_emb = self.raw_static_scfm_gene_emb.to(device)
            emb = self.scfm_gene_projector(raw_emb)
            emb = emb.unsqueeze(0).repeat(num_graphs, 1, 1)
            return emb.reshape(num_graphs * self.num_genes, -1)

        if self.gene_embedding_mode == "scfm_contextual":
            if self.adapter is None:
                raise ValueError("scfm_contextual requires an adapter.")

            x = data.x.reshape(num_graphs, self.num_genes, -1).squeeze(-1).to(device)

            raw_emb = self.adapter.get_contextual_gene_embeddings(x=x, gene_list=self.gene_list)
            if raw_emb is None:
                raise ValueError("Adapter returned None for contextual gene embeddings.")
            if not torch.is_tensor(raw_emb):
                raw_emb = torch.tensor(raw_emb)
            if raw_emb.shape[0] != num_graphs or raw_emb.shape[1] != self.num_genes:
                raise ValueError(
                    f"contextual gene embedding should have shape [{num_graphs}, {self.num_genes}, scfm_dim], but got {raw_emb.shape}"
                )

            raw_emb = raw_emb.to(device).float()
            emb = self.scfm_gene_projector(raw_emb)
            return emb.reshape(num_graphs * self.num_genes, -1)

        raise ValueError(f"Unknown gene_embedding_mode: {self.gene_embedding_mode}")

    def initialize_pert_embeddings_from_scfm(self, raw_pert_emb, freeze_pert_emb=False):
        if raw_pert_emb is None:
            raise ValueError("raw_pert_emb is None.")

        if not torch.is_tensor(raw_pert_emb):
            raw_pert_emb = torch.tensor(raw_pert_emb)

        device = self.args["device"]
        raw_pert_emb = raw_pert_emb.to(device).float()

        if raw_pert_emb.shape[0] != self.num_perts:
            raise ValueError(
                f"raw_pert_emb first dim should be num_perts={self.num_perts}, but got {raw_pert_emb.shape[0]}"
            )

        if raw_pert_emb.shape[1] != self.hidden_size:
            raise ValueError(
                f"raw_pert_emb dim must be hidden_size={self.hidden_size}, but got {raw_pert_emb.shape[1]}. "
                "Please project perturbation embeddings to GEARS hidden_size inside the adapter."
            )

        if raw_pert_emb.shape != self.pert_emb.weight.shape:
            raise ValueError(
                f"raw_pert_emb shape {raw_pert_emb.shape} does not match self.pert_emb.weight shape {self.pert_emb.weight.shape}"
            )

        with torch.no_grad():
            self.pert_emb.weight.copy_(raw_pert_emb)

        self.pert_emb.weight.requires_grad = not freeze_pert_emb

    def forward(self, data):
        x, pert_idx = data.x, data.pert_idx

        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)

        num_graphs = len(data.batch.unique())

        emb = self.get_gene_embedding_for_batch(data, num_graphs)
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)

        gene_ids = torch.arange(self.num_genes, device=self.args["device"]).repeat(num_graphs)
        pos_emb = self.emb_pos(gene_ids)

        for idx, layer in enumerate(self.layers_emb_pos):
            pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.layers_emb_pos) - 1:
                pos_emb = pos_emb.relu()

        base_emb = base_emb + 0.2 * pos_emb
        base_emb = self.emb_trans_v2(base_emb)

        pert_index = []
        for idx, i in enumerate(pert_idx):
            for j in i:
                if j != -1:
                    pert_index.append([idx, j])

        if len(pert_index) > 0:
            pert_index = torch.tensor(pert_index, dtype=torch.long, device=self.args["device"]).T
        else:
            pert_index = torch.empty((2, 0), dtype=torch.long, device=self.args["device"])

        pert_ids = torch.arange(self.num_perts, device=self.args["device"])
        pert_global_emb = self.pert_emb(pert_ids)

        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()

        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

        if pert_index.shape[1] != 0:
            pert_track = {}

            for i, j in enumerate(pert_index[0]):
                cell_idx = j.item()
                pert_gene_idx = pert_index[1][i]

                if cell_idx in pert_track:
                    pert_track[cell_idx] = pert_track[cell_idx] + pert_global_emb[pert_gene_idx]
                else:
                    pert_track[cell_idx] = pert_global_emb[pert_gene_idx]

            if len(list(pert_track.values())) > 0:
                if len(list(pert_track.values())) == 1:
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                else:
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                for idx, j in enumerate(pert_track.keys()):
                    base_emb[j] = base_emb[j] + emb_total[idx]

        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        base_emb = self.bn_pert_base(base_emb)

        base_emb = self.transform(base_emb)
        out = self.recovery_w(base_emb)

        out = out.reshape(num_graphs, self.num_genes, -1)
        out = out.unsqueeze(-1) * self.indv_w1
        w = torch.sum(out, axis=2)
        out = w + self.indv_b1

        cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)
        cross_gene_embed = cross_gene_embed.reshape(num_graphs, self.num_genes, -1)

        cross_gene_out = torch.cat([out, cross_gene_embed], 2)
        cross_gene_out = cross_gene_out * self.indv_w2
        cross_gene_out = torch.sum(cross_gene_out, axis=2)

        out = cross_gene_out + self.indv_b2
        out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
        out = torch.split(torch.flatten(out), self.num_genes)

        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return torch.stack(out), torch.stack(out_logvar)

        return torch.stack(out)
