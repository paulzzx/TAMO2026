import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import retrieval_via_pcst

import datasets
import json

from src.global_path import global_path

model_name = 'sbert'
splits = ["train", "test", "validation"]


class FetaQADataset(Dataset):
    def __init__(self, type, prompt_type='tablellama'):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        self.path = f'{global_path}/dataset/fetaqa'
        self.datas = datasets.load_from_disk(self.path + '/fetaqa')
        self.instruction = 'This is a free-form table question answering task. The goal for this task is to answer the given question based on the given table and the highlighted cells.'
        self._get_linear_table()

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.datas[self.type])

    def _get_linear_table(self, ):
        self.input_seg = []
        self.questions = []
        for data in self.datas[self.type]:
            linear_table = ''
            linear_table += '[TLE] The Wikipedia page title of this table is ' + data[
                'table_page_title'] + '. The Wikipedia section title of this table is ' + data[
                                'table_section_title'] + '. '
            rows = []
            rows.append('[TAB] | ' + ' | '.join(data['table_array'][0]))
            for row in data['table_array'][1:]:
                rows.append('[SEP] | ' + ' | '.join(row))
            linear_table += ' | '.join(rows)
            self.input_seg.append(linear_table)

            highlighted_cells = ['[' + data['table_array'][i][j] + ']' for (i, j) in data['highlighted_cell_ids']]
            question = 'The highlighted cells of the table are: ' + '[HIGHLIGHTED_BEGIN] ' + ', '.join(
                highlighted_cells) + ' [HIGHLIGHTED_END]' + ' ' + data["question"]
            self.questions.append(question)

    def __getitem__(self, index):
        if self.prompt_type == 'tablellama':
            data = self.datas[self.type][index]
            question = f'### Question:\n{self.questions[index].strip()}\n\n### Response:'
            label = data['answer']
            graph = torch.load(f'{self.path}/{self.type}/graphs/{index}.pt')
            desc = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{self.instruction}\n\n### Input:\n{self.input_seg[index]}\n\n"
        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            data = self.datas[self.type][index]
            question = f'### Question:\n{self.questions[index].strip()}\n\n### Response:\n'
            label = data['answer']
            graph = torch.load(f'{self.path}/{self.type}/graphs/{index}.pt')
            desc = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{self.instruction}\n\n### Input:\n{self.input_seg[index]}\n\n"

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }


class FetaQADatasetNohc(Dataset):
    def __init__(self, type):
        super().__init__()
        self.type = type

        self.path = f'{global_path}/dataset/fetaqa'
        self.datas = datasets.load_from_disk(self.path + '/fetaqa')
        self.instruction = 'This is a free-form table question answering task. The goal for this task is to answer the given question based on the given table and the highlighted cells.'
        self._get_linear_table()

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.datas[self.type])

    def _get_linear_table(self, ):
        self.input_seg = []
        for data in self.datas[self.type]:
            linear_table = ''
            linear_table += '[TLE] The Wikipedia page title of this table is ' + data[
                'table_page_title'] + '. The Wikipedia section title of this table is ' + data[
                                'table_section_title'] + '. '
            rows = []
            rows.append('[TAB] | ' + ' | '.join(data['table_array'][0]))
            for row in data['table_array'][1:]:
                rows.append('[SEP] | ' + ' | '.join(row))
            linear_table += ' | '.join(rows)
            self.input_seg.append(linear_table)

    def __getitem__(self, index):
        data = self.datas[self.type][index]
        question = f'### Question:\n{data["question"].strip()}\n\n### Response:'
        label = data['answer']
        graph = torch.load(f'{self.path}/{self.type}/cached_graphs/{index}.pt')
        desc = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{self.instruction}\n\n### Input:\n{self.input_seg[index]}\n\n"

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }
