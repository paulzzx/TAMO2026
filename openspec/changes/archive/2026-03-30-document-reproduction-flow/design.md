## Context

The current repository exposes the main experiment flow through:
- dependency install scripts for server and local environments
- dataset/model download scripts
- preprocessing scripts under `src/dataset/preprocess`
- train and inference entrypoints in `table_train.py` and `inference.py`
- example experiment scripts under `script/`
- smoke tests for local and server validation

The missing piece is not functionality. The missing piece is an explicit, ordered workflow.

From inspection, the workflow naturally splits into two tracks:

```text
                    ┌──────────────────────────────┐
                    │   Reproduce TAMO Project     │
                    └──────────────┬───────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 ▼
        ┌────────────────────┐            ┌────────────────────┐
        │ Local smoke / dev  │            │ Linux/CUDA results │
        └─────────┬──────────┘            └─────────┬──────────┘
                  │                                 │
                  ▼                                 ▼
        requirements.local.txt             requirements.sh
        smoke_test_local.py                download models + datasets
        pure-text path first               preprocess graph artifacts
                                            run train / inference scripts
```

The full TAMO path also splits again:

```text
raw HF/local dataset
        │
        ▼
load_from_disk(...)
        │
        ├── pure-text baseline
        │     table_train.py / inference.py
        │     model_name in {llm, pt_llm, mistral, pt_mistral}
        │
        └── TAMO hypergraph
              preprocess/*.py
              ├── graphs/<index>.pt
              ├── some datasets also write q_embs.pt
              └── table_train.py / inference.py with table_hypergraph_*
```

## Goals / Non-Goals

**Goals:**
- Record one ordered workflow for minimum validation and one for paper-style experiment reproduction.
- Make explicit which files prove each step is complete.
- Capture repository mismatches that can break a naive reproduction attempt.
- Provide enough structure that a future implementation change can turn this into README or automation work.

**Non-Goals:**
- Fix repository inconsistencies in this change.
- Rewrite shell scripts.
- Add automation, CI, or one-click experiment launchers.
- Guarantee exact paper metrics without the original training environment and checkpoints.

## Decisions

1. Record two levels of reproduction instead of one.

The repository already supports both lightweight local validation and full experiment execution. Mixing them into one path would obscure the real dependency boundary:
- local validation can run with `requirements.local.txt` and mocked model objects
- paper-style reproduction depends on Linux/CUDA, local model weights, PyG compiled extensions, and graph preprocessing artifacts

2. Treat preprocessing as a hard prerequisite for hypergraph runs.

The dataset classes for `structprobe`, `wtq_orig`, `wikisql`, `fetaqa`, and `hitab` all load graph artifacts from `<dataset>/<split>/graphs/<index>.pt`. Without preprocessing, TAMO model variants will fail even if the raw datasets are present.

3. Use smoke tests as the first validation gates.

The smoke tests define the practical minimum success criteria:
- `smoke_test_local.py` validates local dataset loading, graph collation, and a mocked pure-text model path
- `server_smoke_test.py` validates imports, CUDA, PyG extensions, local dataset presence, optional graph artifacts, and optional real model loading

4. Record known mismatches as reproduction risks, not as hidden assumptions.

Current mismatches discovered during inspection:
- `README.md` refers to `data_preprocesss.sh`, but the actual script is `script/data_preprocess.sh`
- `src/model/__init__.py` expects `models/meta-llama/Llama-2-7b-chat-hf`, but `download_models.sh` downloads into `models/meta-llama/Llama-2-7b-chat`
- several `llama2_*.sh` scripts actually launch Mistral prompt/model combinations
- the repository currently includes HF datasets on disk, but public-benchmark download and preprocessing are still part of the formal workflow

## Risks / Trade-offs

- A documented workflow can still drift if the shell scripts or path conventions change later.
- Exact numerical reproduction may still depend on hardware, CUDA stack, and external model access permissions.
- The workflow can clarify the order of operations, but it cannot remove the operational cost of downloading gated base models such as Llama 2.
