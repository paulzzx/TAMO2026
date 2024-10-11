from torch_geometric.data import Data
import torch


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_index = edge_index  # [2, N]
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'corr_edge_index', 'edge_index_corr1', 'edge_index_corr2']:
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
