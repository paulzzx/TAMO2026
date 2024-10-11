import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import retrieval_via_pcst

import datasets

from src.global_path import global_path
import random

model_name = 'sbert'
splits = ["train", "test", "validation"]


class WTQDatasetOrig(Dataset):
    def __init__(self, type, prompt_type='tablellama'):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        self.path = f'{global_path}/dataset/wtq'
        self.datas = datasets.load_from_disk(self.path + '/wikitablequestions')
        self.instruction = 'This is a table QA task. The goal of this task is to answer the question given the table.'
        self._get_linear_table()

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.datas[self.type])

    # def _get_linear_table(self,):
    #     self.input_seg = []
    #     for data in self.datas[self.type]:
    #         table_array = []
    #         table_array.append(data["table"]["header"])
    #         table_array.extend(data["table"]["rows"])
    #         linear_table = ''
    #         linear_table += 'col : '+' | '.join(table_array[0])
    #         for idx,row in enumerate(table_array[1:]):
    #             linear_table += ' row ' + str(idx+1) +' : '+' | '.join(row)
    #         self.input_seg.append(linear_table)

    def _get_linear_table(self, ):
        self.input_seg = []
        for data in self.datas[self.type]:
            table_array = []
            table_array.append(data["table"]["header"])
            table_array.extend(data["table"]["rows"])

            rows = []
            rows.append('[TAB] col : ' + ' | '.join(table_array[0]))
            for row in table_array[1:]:
                rows.append('[SEP] | ' + ' | '.join(row))
            linear_table = ' | '.join(rows)
            self.input_seg.append(linear_table)

    def __getitem__(self, index):
        if self.prompt_type == 'tablellama':
            data = self.datas[self.type][index]
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = ', '.join(data['answers'])
            graph = torch.load(f'{self.path}/{self.type}/graphs/{index}.pt')
            desc = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{self.instruction}\n\n### Input:\n{self.input_seg[index]}\n\n"
        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            data = self.datas[self.type][index]
            question = f'### Question:\n{data["question"]}\n\n### Response:\n'
            label = ', '.join(data['answers'])
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
