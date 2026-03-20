import re
import os
import torch
import pandas as pd
import random
import ast

from tqdm import tqdm
from torch_geometric.data.data import Data
from src.dataset.utils.graph_data import BipartiteData

from src.utils.lm_modeling import load_model, load_text2embedding

import datasets

from src.global_path import global_path

model_name = 'sbert'

path = f'{global_path}/dataset/hitab'
datas = datasets.load_from_disk(path + '/hitab')
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
        if {'src': nodes[src], 'dst': hyperedges[dst]} not in edges:
            edges.append({'src': nodes[src], 'dst': hyperedges[dst]})

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    hyperedges = pd.DataFrame(hyperedges.items(), columns=['hyperedge_attr', 'hyperedge_id'])
    edges = pd.DataFrame(edges)

    return nodes, hyperedges, edges


def dfs(root, floor, type, hi_dict, path):
    if root['row_index'] >= 0 and root['column_index'] >= 0:
        path.append((root['row_index'], root['column_index']))

    if not root['children']:
        hi_dict[(root['row_index'], root['column_index'])] = path
        if type == 'top':
            floor[root['column_index']] = root['row_index']
        else:
            floor[root['row_index']] = root['column_index']

    for child in root['children']:
        dfs(child, floor, type, hi_dict, path.copy())


def fill_table_cell(table_text, merged_regions):
    for merged_region in merged_regions:
        cell_value = table_text[merged_region['first_row']][merged_region['first_column']]
        for i in range(merged_region['first_row'], merged_region['last_row'] + 1):
            for j in range(merged_region['first_column'], min(len(table_text[0]), merged_region['last_column'] + 1)):
                table_text[i][j] = cell_value


def convert_table_2_triplets(table):
    table = ast.literal_eval(table)
    row_floor = {}
    column_floor = {}
    hi_dict = {}
    dfs(table['top_root'], row_floor, 'top', hi_dict, [])
    dfs(table['left_root'], column_floor, 'left', hi_dict, [])

    row, column = [], []
    for key, value in row_floor.items():
        column.append(key)
    for key, value in column_floor.items():
        row.append(key)

    table_texts = table['texts']
    fill_table_cell(table_texts, table['merged_regions'])
    triplets = []
    for i in row:
        for j in column:
            cell = table_texts[i][j]
            #### column hyperedge ####
            column_hyperedges = (row_floor[j], j)
            for value in hi_dict[column_hyperedges]:
                triplets.append((cell, 'belong to this column', table_texts[value[0]][value[1]]))
            #### row hyperedge ####
            row_hyperedges = (i, column_floor[i])
            for value in hi_dict[row_hyperedges]:
                triplets.append((cell, 'belong to this row', table_texts[value[0]][value[1]]))

    return triplets


def step_one():
    # generate textual graphs
    for sp in splits:
        for i, data in tqdm(enumerate(datas[sp]), desc=f'step one {sp}: '):
            nodes, hyperedges, edges = textualize_graph(convert_table_2_triplets(data['table_content']))
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
            print("x_s.shape:", x_s.shape)
            print("len(nodes):", len(nodes))
            assert x_s.shape[0] == len(nodes)
            assert x_t.shape[0] == len(hyperedges)
            data = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
            torch.save(data, f'{path}/{sp}/graphs/{i}.pt')

    def _encode_questions(sp):
        print(f'Encoding questions for [{sp}]...')
        q_embs = text2embedding(model, tokenizer, device, datas[sp]['question'])
        torch.save(q_embs, f'{path}/{sp}/q_embs.pt')

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    for sp in splits:
        _encode_graph(sp)
        # _encode_questions(sp)


if __name__ == '__main__':
    step_one()
    step_two()
