## Why

The repository already contains the core training, inference, preprocessing, and smoke-test entrypoints for TAMO, but the end-to-end reproduction workflow is still implicit. A new contributor can find the scripts, but not the exact order, prerequisites, or the difference between a local smoke run, a pure-text baseline, and the full TAMO hypergraph pipeline.

This change records a concrete reproduction workflow so the project can be reproduced with fewer false starts and with clearer expectations about datasets, model assets, graph preprocessing, and output locations.

## What Changes

Document the reproduction workflow for TAMO experiments as an OpenSpec capability.

The recorded workflow should:
- distinguish local debugging from Linux/CUDA experiment reproduction
- distinguish pure-text baselines from TAMO hypergraph runs
- define the required order: environment setup, dataset/model download, preprocessing, smoke test, training or inference, result inspection
- capture known repository inconsistencies that affect reproduction

## Capabilities

### New Capabilities
- `reproduction-workflow`: Define the required inputs, steps, and validation points for reproducing TAMO locally and on a training server.

### Modified Capabilities
- None.

## Impact

Affected areas:
- `README.md`
- `requirements.sh`
- `requirements.local.txt`
- `download_dataset.py`
- `download_models.sh`
- `script/*.sh`
- `table_train.py`
- `inference.py`
- `smoke_test_local.py`
- `server_smoke_test.py`

This change records process knowledge only. It does not implement or modify the application code.
