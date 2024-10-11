import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import retrieval_via_pcst
import ast

import datasets

from src.global_path import global_path
import random

model_name = 'sbert'
splits = ["train", "test", "validation"]


class HiTabDataset(Dataset):
    def __init__(self, type, prompt_type='tablellama'):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        self.path = f'{global_path}/dataset/hitab'
        self.datas = datasets.load_from_disk(self.path + '/hitab')
        self.instruction = 'This is a hierarchical table question answering task. The goal for this task is to answer the given question based on the given table. The table might be hierarchical.'
        self._get_linear_table()

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.datas[self.type])

    def _fill_table_cell(self, table_text, merged_regions):
        for merged_region in merged_regions:
            cell_value = table_text[merged_region['first_row']][merged_region['first_column']]
            for i in range(merged_region['first_row'], merged_region['last_row'] + 1):
                for j in range(merged_region['first_column'],
                               min(len(table_text[0]), merged_region['last_column'] + 1)):
                    table_text[i][j] = cell_value
        return table_text

    def _get_linear_table(self, ):
        self.input_seg = []
        for data in self.datas[self.type]:
            table_content = ast.literal_eval(data['table_content'])
            table_array = self._fill_table_cell(table_content['texts'], table_content['merged_regions'])

            rows = []
            rows.append('[TAB] | ' + ' | '.join(table_array[0]))
            for row in table_array[1:]:
                rows.append('[SEP] | ' + ' | '.join(row))
            linear_table = ' | '.join(rows)
            linear_table = '[TLE] The table caption is ' + table_content['title'] + '. ' + linear_table
            self.input_seg.append(linear_table)

    def __getitem__(self, index):
        data = self.datas[self.type][index]
        data['answer'] = ast.literal_eval(data['answer'])
        data['answer'] = [str(i) for i in data['answer']]
        if self.prompt_type == 'tablellama':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = ', '.join(data['answer'])
            graph = torch.load(f'{self.path}/{self.type}/graphs/{index}.pt')
            desc = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{self.instruction}\n\n### Input:\n{self.input_seg[index]}\n\n"
        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            question = f'### Question:\n{data["question"]}\n\n### Response:\n'
            label = ', '.join(data['answer'])
            graph = torch.load(f'{self.path}/{self.type}/graphs/{index}.pt')
            desc = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n### Instruction:\n{self.instruction}\n\n### Input:\n{self.input_seg[index]}\n\n"
        else:
            raise ValueError(f'prompt_type {self.prompt_type} is not supported')

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }
