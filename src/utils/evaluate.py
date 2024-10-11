import json
import pandas as pd
import re
import string

from collections import defaultdict
from typing import List

from datasets import load_metric

def get_accuracy_wtq(eval_output, path=None, input_df=None):
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])
    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    delimiter = ", "
    def evaluate_example(predict_str: str, ground_str: str):
        predict_spans = predict_str.split(delimiter)
        ground_spans = ground_str.split(delimiter)
        predict_values = defaultdict(lambda: 0)
        ground_values = defaultdict(lambda: 0)
        for span in predict_spans:
            try:
                predict_values[float(span)] += 1
            except ValueError:
                predict_values[span.strip()] += 1
        for span in ground_spans:
            try:
                ground_values[float(span)] += 1
            except ValueError:
                ground_values[span.strip()] += 1
        _is_correct = predict_values == ground_values
        return _is_correct
    
    def get_denotation_accuracy(predictions: List[str], references: List[str]):
        assert len(predictions) == len(references)
        correct_num = 0
        for predict_str, ground_str in zip(predictions, references):
            is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
            if is_correct:
                correct_num += 1
        return correct_num / len(predictions)
    
    acc = get_denotation_accuracy(df["pred"].tolist(), df["label"].tolist())

    return acc

def get_blue(eval_output, path=None, input_df=None):
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])
    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")
    
    preds,labels = df["pred"].tolist(), df["label"].tolist()
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    sacrebleu_metric = load_metric('sacrebleu', lowercase=True, trust_remote_code=True)
    sacrebleu_preds = preds
    sacrebleu_labels = [[label] for label in labels]
    sacrebleu_res = sacrebleu_metric.compute(predictions=sacrebleu_preds, references=sacrebleu_labels)

    return sacrebleu_res['score']


def get_accuracy_tabfact(eval_output, path=None, input_df=None):
    if input_df is not None:
        df = input_df
    else:
        df = pd.concat([pd.DataFrame(d) for d in eval_output])
    if path:
        with open(path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")

    delimiter = ", "
    def evaluate_example(predict_str: str, ground_str: str):
        predict_str = predict_str.split('.')[0]

        predict_spans = predict_str.split(delimiter)
        ground_spans = ground_str.split(delimiter)
        predict_values = defaultdict(lambda: 0)
        ground_values = defaultdict(lambda: 0)
        for span in predict_spans:
            try:
                predict_values[float(span)] += 1
            except ValueError:
                predict_values[span.strip()] += 1
        for span in ground_spans:
            try:
                ground_values[float(span)] += 1
            except ValueError:
                ground_values[span.strip()] += 1
        _is_correct = predict_values == ground_values
        return _is_correct
    
    def get_denotation_accuracy(predictions: List[str], references: List[str]):
        assert len(predictions) == len(references)
        correct_num = 0
        for predict_str, ground_str in zip(predictions, references):
            is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
            if is_correct:
                correct_num += 1
        return correct_num / len(predictions)
    
    acc = get_denotation_accuracy(df["pred"].tolist(), df["label"].tolist())

    return acc



eval_funcs = {
    'wtq_orig' : get_accuracy_wtq,
    'wtq_permute' : get_accuracy_wtq,
    'wikisql' : get_accuracy_wtq,
    'fetaqa' : get_blue,
    'fetaqa_nohc' : get_blue,
    'structprobe': get_accuracy_wtq,
    'structprobe_permute': get_accuracy_wtq,
    'hitab': get_accuracy_wtq,
}
