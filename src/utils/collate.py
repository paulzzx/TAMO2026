from torch_geometric.data import Batch
from torch_geometric.loader.dataloader import Collater


def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
        # batch['graph'] = Collater(batch['graph'])
    return batch
