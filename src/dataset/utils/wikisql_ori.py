# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import tarfile
from copy import deepcopy
import requests

from tqdm import tqdm

from src.dataset.utils.wikisql_executor import retrieve_wikisql_query_answer_tapas, _TYPE_CONVERTER

RAW_DATASET_FOLDER = "raw_dataset"
PROCESSED_DATASET_FOLDER = "dataset"


def download_file(url, download_dir=None):
    """
    Download file into local file system from url
    """
    local_filename = url.split('/')[-1]
    if download_dir is None:
        download_dir = os.curdir
    elif not os.path.exists(download_dir):
        os.makedirs(download_dir)
    with requests.get(url, stream=True) as r:
        file_name = os.path.join(download_dir, local_filename)
        if os.path.exists(file_name):
            os.remove(file_name)
        write_f = open(file_name, "wb")
        for data in tqdm(r.iter_content()):
            write_f.write(data)
        write_f.close()

    return os.path.abspath(file_name)


def download_wikisql():
    """
    Download WikiSQL dataset and unzip the files
    """
    wikisql_url = "https://raw.github.com/salesforce/WikiSQL/master/data.tar.bz2"
    wikisql_raw_path = os.path.join(RAW_DATASET_FOLDER, "wikisql")
    wikisql_tar_file = download_file(wikisql_url)
    # unzip and move it into raw_dataset folder
    tar = tarfile.open(wikisql_tar_file, "r:bz2")
    tar.extractall(wikisql_raw_path)
    tar.close()
    # remove the original file
    os.remove(wikisql_tar_file)
    return wikisql_raw_path


def build_wikisql_fariseq_dataset(out_prefix, src_file, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def _convert_table_types(table):
        """Runs the type converter over the table cells."""
        ret_table = deepcopy(table)
        types = ret_table['types']
        ret_table['real_rows'] = ret_table['rows']
        typed_rows = []
        for row in ret_table['rows']:
            typed_row = []
            for column, cell_value in enumerate(row):
                typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
            typed_rows.append(typed_row)
        ret_table['rows'] = typed_rows
        return ret_table

    # load table content dictionary from files
    table_content_dict = {}
    table_file_path = "{}.tables.jsonl".format(src_file.split(".")[0])
    for json_line in open(table_file_path, "r", encoding="utf8"):
        content = json.loads(json_line)
        table_content_dict[content["id"]] = content

    examples = open(src_file, "r", encoding="utf8").readlines()
    for example in examples:
        # each line is a json object
        example = json.loads(example)
        table_id = example["table_id"]
        table_content = table_content_dict[table_id]
        question = example["question"].lower()
        tapas_table = _convert_table_types(table_content)
        # retrieve wikisql answers by TaPaS script as ground-truth and training labels
        answer = retrieve_wikisql_query_answer_tapas(tapas_table, example)


if __name__ == '__main__':
    wikisql_raw_data_dir = 'raw_dataset/wikisql'
    processed_wikisql_data_dir = os.path.join(PROCESSED_DATASET_FOLDER, "wikisql")
    build_wikisql_fariseq_dataset("train", os.path.join(wikisql_raw_data_dir, "data", "train.jsonl"),
                                  processed_wikisql_data_dir)
    build_wikisql_fariseq_dataset("valid", os.path.join(wikisql_raw_data_dir, "data", "dev.jsonl"),
                                  processed_wikisql_data_dir)
    build_wikisql_fariseq_dataset("test", os.path.join(wikisql_raw_data_dir, "data", "test.jsonl"),
                                  processed_wikisql_data_dir)
