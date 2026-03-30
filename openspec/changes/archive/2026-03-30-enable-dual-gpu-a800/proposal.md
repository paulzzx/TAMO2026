## Why

The current TAMO experiment surface is mostly single-GPU by default, with most scripts hard-coding `CUDA_VISIBLE_DEVICES=0`. That does not match the target execution environment, where experiments should use two A800 80G GPUs (`GPU0` and `GPU1`) to increase available memory headroom and reduce out-of-memory risk for large model runs.

This change proposes a consistent dual-GPU execution setup so the repository's training and inference commands align with the intended hardware baseline instead of depending on one-off manual edits before every run.

## What Changes

- Standardize experiment launch scripts around dual-GPU execution on `GPU0,GPU1`.
- Define which scripts should run on both GPUs and which preprocessing steps can remain single-GPU.
- Record the expected runtime behavior for train and inference entrypoints when two GPUs are exposed.
- Update reproduction-facing guidance so the A800 80G dual-GPU setup is explicit.
- Add any lightweight runtime or script-level adjustments needed to make dual-GPU launches consistent.

## Capabilities

### New Capabilities
- `dual-gpu-execution`: Define how TAMO experiments are launched and validated on dual A800 80G GPUs using `GPU0` and `GPU1`.

### Modified Capabilities
- None.

## Impact

Affected areas:
- `script/*.sh`
- `README.md`
- train and inference launch conventions
- optional startup validation around visible GPUs

This change is about execution configuration and reproducible launch behavior, not model redesign.
