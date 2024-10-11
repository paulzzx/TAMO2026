import re
import os
import torch
import pandas as pd
import random

from tqdm import tqdm
from torch_geometric.data.data import Data
from src.dataset.utils.graph_data import BipartiteData

from src.utils.lm_modeling import load_model, load_text2embedding

import datasets

from src.global_path import global_path

model_name = 'sbert'

path = f'{global_path}/dataset/wtq'
datas = datasets.load_from_disk(path + '/wikitablequestions')
splits = ["train", "test", "validation"]


def textualize_graph(graph):
    triplets = graph
    nodes = {}
    hyperedges = {}
    edges = []
    for tri in triplets:
        src, edeg_attr, dst = tri
        src = src.lower().strip()
        dst = dst.lower().strip()
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in hyperedges:
            hyperedges[dst] = len(hyperedges)
        edges.append({'src': nodes[src], 'dst': hyperedges[dst]})

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    hyperedges = pd.DataFrame(hyperedges.items(), columns=['hyperedge_attr', 'hyperedge_id'])
    edges = pd.DataFrame(edges)

    return nodes, hyperedges, edges


def convert_table_2_triplets(header, rows):
    triplets = []
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            triplets.append((cell, 'belong to this row', f'row {i + 1}'))
            triplets.append((cell, 'belong to this column', header[j]))

    return triplets


def step_one():
    # generate textual graphs
    for sp in splits:
        for i, data in tqdm(enumerate(datas[sp]), desc=f'step one {sp}: '):
            nodes, hyperedges, edges = textualize_graph(
                convert_table_2_triplets(data['table']['header'], data['table']['rows']))
            os.makedirs(f'{path}/{sp}/nodes', exist_ok=True)
            os.makedirs(f'{path}/{sp}/hyperedges', exist_ok=True)
            os.makedirs(f'{path}/{sp}/edges', exist_ok=True)
            nodes.to_csv(f'{path}/{sp}/nodes/{i}.csv', index=False, columns=['node_attr', 'node_id'])
            hyperedges.to_csv(f'{path}/{sp}/hyperedges/{i}.csv', index=False,
                              columns=['hyperedge_attr', 'hyperedge_id'])
            edges.to_csv(f'{path}/{sp}/edges/{i}.csv', index=False, columns=['src', 'dst'])


def step_two():
    def _encode_graph(sp):
        print(f'Encoding graphs for [{sp}]...')
        os.makedirs(f'{path}/{sp}/graphs', exist_ok=True)
        for i in tqdm(range(len(datas[sp]))):
            nodes = pd.read_csv(f'{path}/{sp}/nodes/{i}.csv').astype(str)
            hyperedges = pd.read_csv(f'{path}/{sp}/hyperedges/{i}.csv').astype(str)
            edges = pd.read_csv(f'{path}/{sp}/edges/{i}.csv')
            nodes, hyperedges, edges = nodes.fillna('null node'), hyperedges.fillna('null hyperedge'), edges.fillna(
                'null edge')
            x_s = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            x_t = text2embedding(model, tokenizer, device, hyperedges.hyperedge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])
            assert x_s.shape[0] == len(nodes)
            assert x_t.shape[0] == len(hyperedges)
            data = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
            torch.save(data, f'{path}/{sp}/graphs/{i}.pt')

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    for sp in splits:
        _encode_graph(sp)


if __name__ == '__main__':
    step_one()
    step_two()
