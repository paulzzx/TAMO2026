#!/usr/bin/env python3
"""Local smoke test for TAMO on macOS.

This script is intentionally lightweight:
- it does not download any real model weights
- it does not require Linux CUDA wheels
- it validates the main local code paths with synthetic stand-ins

Covered paths:
1) local dataset can be loaded from disk
2) graph samples can be batched with the project collate function
3) the pure-text LLM path can be instantiated and run with mocked HF objects
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def check_dataset() -> None:
    import datasets

    dataset_path = Path("dataset/structProbe/structProbe")
    assert dataset_path.exists(), f"missing dataset path: {dataset_path}"

    ds = datasets.load_from_disk(str(dataset_path))
    expected_splits = {"train", "validation", "test"}
    assert expected_splits.issubset(set(ds.keys())), f"unexpected splits: {list(ds.keys())}"

    sample = ds["train"][0]
    for key in ("question", "answers", "table"):
        assert key in sample, f"dataset sample missing key: {key}"
    assert "header" in sample["table"] and "rows" in sample["table"], "table schema mismatch"

    print("dataset: ok")
    print(f"  train size: {len(ds['train'])}")
    print(f"  sample question: {sample['question'][:80]!r}")


def check_graph_collate() -> None:
    import torch

    from src.dataset.utils.graph_data import BipartiteData
    from src.utils.collate import collate_fn

    graph_a = BipartiteData(
        x_s=torch.randn(3, 4),
        x_t=torch.randn(2, 4),
        edge_index=torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long),
    )
    graph_b = BipartiteData(
        x_s=torch.randn(2, 4),
        x_t=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
    )

    batch = collate_fn(
        [
            {"id": 0, "question": "q0", "label": "a0", "desc": "d0", "graph": graph_a},
            {"id": 1, "question": "q1", "label": "a1", "desc": "d1", "graph": graph_b},
        ]
    )

    assert batch["graph"]._num_graphs == 2, "graph batch size mismatch"
    assert batch["graph"].x_s.size(0) == 5, "source node concat mismatch"
    assert batch["graph"].x_t.size(0) == 5, "target node concat mismatch"

    print("graph collate: ok")


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        import torch

        def encode_one(s: str) -> list[int]:
            base = [((ord(ch) % 31) + 3) for ch in s[:32]]
            return base or [3]

        if isinstance(text, list):
            ids = [encode_one(x) for x in text]
            if return_tensors == "pt":
                max_len = max(len(x) for x in ids)
                padded = [[0] * (max_len - len(x)) + x for x in ids]
                mask = [[0] * (max_len - len(x)) + [1] * len(x) for x in ids]
                return SimpleNamespace(
                    input_ids=torch.tensor(padded, dtype=torch.long),
                    attention_mask=torch.tensor(mask, dtype=torch.long),
                )
            return SimpleNamespace(input_ids=ids)

        ids = encode_one(text)
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def batch_decode(self, outputs, skip_special_tokens=True):
        return ["decoded output" for _ in range(outputs.size(0))]


class DummyConfig:
    def __init__(self):
        self.max_position_embeddings = 2048
        self.rope_scaling = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class DummyInnerModel:
    def __init__(self, embedding):
        self._embedding = embedding

    def get_input_embeddings(self):
        return self._embedding


class DummyCausalLM:
    def __init__(self):
        import torch

        self.embedding = torch.nn.Embedding(4096, 16)
        self.model = DummyInnerModel(self.embedding)
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def named_parameters(self):
        return self.embedding.named_parameters()

    def __call__(self, inputs_embeds=None, attention_mask=None, return_dict=True, labels=None):
        import torch

        loss = inputs_embeds.float().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("dummy loss is not finite")
        return SimpleNamespace(loss=loss)

    def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=8, **kwargs):
        import torch

        batch_size = inputs_embeds.size(0)
        return torch.full((batch_size, max_new_tokens), 7, dtype=torch.long)


def check_pure_text_model() -> None:
    import torch

    from src.model import load_model

    args = Namespace(
        max_txt_len=64,
        max_new_tokens=8,
        dataset="structprobe",
        llm_frozen="True",
        llm_lora="False",
        do_eval="True",
        llm_model_path="dummy",
        llm_num_virtual_tokens=8,
        num_token=1,
        gnn_num_layers=1,
        gnn_num_heads=1,
        gnn_hidden_dim=16,
        gnn_in_dim=16,
    )

    with patch("transformers.AutoTokenizer.from_pretrained", DummyTokenizer.from_pretrained), patch(
        "transformers.AutoConfig.from_pretrained", DummyConfig.from_pretrained
    ), patch("transformers.AutoModelForCausalLM.from_pretrained", DummyCausalLM.from_pretrained):
        model = load_model["llm"](init_prompt="Please follow the instruction below.", args=args)
        loss = model(
            {
                "id": [0, 1],
                "question": ["### Question:\nq0\n\n### Response:\n", "### Question:\nq1\n\n### Response:\n"],
                "label": ["yes", "no"],
                "desc": ["desc a", "desc b"],
            }
        )
        assert torch.is_tensor(loss), "forward did not return a tensor"
        pred = model.inference(
            {
                "id": [0, 1],
                "question": ["### Question:\nq0\n\n### Response:\n", "### Question:\nq1\n\n### Response:\n"],
                "label": ["yes", "no"],
                "desc": ["desc a", "desc b"],
            }
        )
        assert len(pred["pred"]) == 2, "inference output size mismatch"

    print("pure-text model path: ok")


def main() -> None:
    print("Running TAMO local smoke test...")
    check_dataset()
    check_graph_collate()
    check_pure_text_model()
    print("smoke test: passed")


if __name__ == "__main__":
    main()
