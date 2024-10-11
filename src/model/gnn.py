import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.hyper_layer import AllSetTrans


class HyperGraph(nn.Module):
    def __init__(self, config):
        super(HyperGraph, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, data):
        embedding_s, embedding_t = data.x_s, data.x_t
        embedding_t = torch.cat([embedding_t, embedding_s], dim=0)

        # Add self-loop
        num_nodes, num_hyper_edges = data.x_s.size(0), data.x_t.size(0)
        self_edge_index = torch.tensor([[i, num_hyper_edges + i] for i in range(num_nodes)]).T
        if ('edge_neg_view' in self.config.to_dict() and self.config.edge_neg_view == 1):
            edge_index = torch.cat([data.edge_index_corr1, self_edge_index.to(data.edge_index_corr1.device)], dim=-1)
        elif ('edge_neg_view' in self.config.to_dict() and self.config.edge_neg_view == 2):
            edge_index = torch.cat([data.edge_index_corr2, self_edge_index.to(data.edge_index_corr2.device)], dim=-1)
        else:
            edge_index = torch.cat([data.edge_index, self_edge_index.to(data.edge_index.device)], dim=-1)

        for i, layer_module in enumerate(self.layer):
            embedding_s, embedding_t = layer_module(embedding_s, embedding_t, edge_index)
        outputs = (embedding_s, embedding_t[:num_hyper_edges])

        return outputs


class EncoderLayer(nn.Module):
    """SetTransformer Encoder Layer"""

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.V2E = AllSetTrans(config=config)
        self.fuse = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.E2V = AllSetTrans(config=config)

    def forward(self, embedding_s, embedding_t, edge_index):
        # reverse the index
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        # from nodes to hyper-edges
        embedding_t_tem = F.relu(self.V2E(embedding_s, edge_index))

        # from hyper-edges to nodes
        embedding_t = torch.cat([embedding_t, embedding_t_tem], dim=-1)
        # fuse the output t_embeds with original t_embeds, or the t_embeds will not have the original info
        embedding_t = F.dropout(self.fuse(embedding_t), p=self.dropout, training=self.training)
        embedding_s = F.relu(self.E2V(embedding_t, reversed_edge_index))
        embedding_s = F.dropout(embedding_s, p=self.dropout, training=self.training)

        return embedding_s, embedding_t


load_gnn_model = {
    'hyper': HyperGraph
}
