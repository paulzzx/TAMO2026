#!/usr/bin/env python3
"""Server smoke test for TAMO on Linux/CUDA.

Checks:
1) core package imports and version reporting
2) CUDA visibility from torch
3) PyG extension imports
4) local dataset loading
5) optional graph artifact presence
6) real tokenizer/model loading from local path
7) one tiny real generate call

Environment variables:
- SMOKE_MODEL_NAME: runtime LLM key in src.model.llama_model_path_aliases, default "7b"
- SMOKE_PROMPT_TYPE: default "llama2"
- SMOKE_DATASET: default "structprobe"
- SMOKE_SKIP_MODEL: "1" to skip real model loading
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def check_core_imports() -> None:
    print_header("Core Imports")
    import torch
    import transformers
    import peft
    import datasets
    import numpy
    import pandas
    import pyarrow
    import gensim
    import wandb
    import sentencepiece

    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("peft:", peft.__version__)
    print("datasets:", datasets.__version__)
    print("numpy:", numpy.__version__)
    print("pandas:", pandas.__version__)
    print("pyarrow:", pyarrow.__version__)
    print("gensim:", gensim.__version__)
    print("wandb:", wandb.__version__)
    print("sentencepiece:", sentencepiece.__version__)


def check_cuda() -> None:
    print_header("CUDA")
    import torch

    print("torch.version.cuda:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in torch")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device visible to torch")
    print("device_0:", torch.cuda.get_device_name(0))


def check_pyg_imports() -> None:
    print_header("PyG Imports")
    import torch_geometric
    import torch_scatter
    import torch_sparse
    import torch_cluster
    import torch_spline_conv

    print("torch-geometric:", torch_geometric.__version__)
    print("torch-scatter: ok")
    print("torch-sparse: ok")
    print("torch-cluster: ok")
    print("torch-spline-conv: ok")


def check_dataset(dataset_name: str) -> None:
    print_header("Dataset")
    import datasets

    mapping = {
        "structprobe": Path("dataset/structProbe/structProbe"),
        "wtq_orig": Path("dataset/wtq/wikitablequestions"),
        "wikisql": Path("dataset/wikisql/wikisql"),
        "fetaqa": Path("dataset/fetaqa/fetaqa"),
        "hitab": Path("dataset/hitab/hitab"),
    }
    if dataset_name not in mapping:
        raise ValueError(f"unsupported smoke dataset: {dataset_name}")

    dataset_path = mapping[dataset_name]
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")

    ds = datasets.load_from_disk(str(dataset_path))
    print("dataset path:", dataset_path)
    print("splits:", list(ds.keys()))
    print("train size:", len(ds["train"]))
    sample = ds["train"][0]
    print("sample keys:", sorted(sample.keys()))


def check_graph_artifacts(dataset_name: str) -> None:
    print_header("Graph Artifacts")
    graph_root = {
        "structprobe": Path("dataset/structProbe/train/graphs"),
        "wtq_orig": Path("dataset/wtq/train/graphs"),
        "wikisql": Path("dataset/wikisql/train/graphs"),
        "fetaqa": Path("dataset/fetaqa/train/graphs"),
        "hitab": Path("dataset/hitab/train/graphs"),
    }.get(dataset_name)

    if graph_root is None:
        print("graph check skipped: no graph mapping for dataset")
        return

    if not graph_root.exists():
        print(f"warning: graph directory missing: {graph_root}")
        print("warning: hypergraph training/inference will fail until preprocessing is done")
        return

    pt_files = sorted(graph_root.glob("*.pt"))
    print("graph dir:", graph_root)
    print("graph files:", len(pt_files))
    if not pt_files:
        raise RuntimeError(f"graph directory exists but contains no .pt files: {graph_root}")


def check_real_model(model_name: str, prompt_type: str) -> None:
    print_header("Real Model Load")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.model import ensure_known_llm_key, resolve_llm_model_path

    ensure_known_llm_key(model_name)

    model_path = Path(resolve_llm_model_path(model_name))

    print("model key:", model_name)
    print("model path:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    prompt = (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\nAnswer briefly.\n\n"
        "### Input:\ncol : name | score [SEP] row alice : 10 [SEP]\n\n"
        "### Question:\nWhat is the score for alice?\n\n"
        "### Response:\n"
    )
    if prompt_type == "mistral":
        prompt = "Question: What is the score for alice?\nTable: name | score ; alice | 10\nAnswer:"

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print("generate: ok")
    print("decoded preview:", repr(decoded[:200]))


def main() -> int:
    model_name = os.environ.get("SMOKE_MODEL_NAME", "7b")
    prompt_type = os.environ.get("SMOKE_PROMPT_TYPE", "llama2")
    dataset_name = os.environ.get("SMOKE_DATASET", "structprobe")
    skip_model = os.environ.get("SMOKE_SKIP_MODEL", "0") == "1"

    print("server smoke test config:")
    print("  model:", model_name)
    print("  prompt_type:", prompt_type)
    print("  dataset:", dataset_name)
    print("  skip_model:", skip_model)

    check_core_imports()
    check_cuda()
    check_pyg_imports()
    check_dataset(dataset_name)
    check_graph_artifacts(dataset_name)
    if skip_model:
        print_header("Real Model Load")
        print("skipped by SMOKE_SKIP_MODEL=1")
    else:
        check_real_model(model_name, prompt_type)

    print("\nserver smoke test: passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nserver smoke test: failed: {exc}", file=sys.stderr)
        raise
