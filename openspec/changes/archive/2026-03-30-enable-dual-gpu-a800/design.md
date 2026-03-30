## Context

The repository currently mixes two execution assumptions:
- most scripts run with `CUDA_VISIBLE_DEVICES=0`
- `script/llama2_sft.sh` already assumes two GPUs via `CUDA_VISIBLE_DEVICES=0,1`

The model code itself frequently loads Hugging Face causal LMs with `device_map="auto"`, which means the visible CUDA device set materially changes model placement. In practice, exposing two 80G A800 GPUs allows significantly larger effective memory capacity than the current single-GPU script defaults.

The desired target is not "generic multi-GPU someday". It is a concrete deployment baseline:

```text
Machine: dual A800 80G
Visible GPUs for TAMO runs: GPU0, GPU1
CUDA_VISIBLE_DEVICES=0,1
```

This change needs to make that baseline explicit and consistent.

## Goals / Non-Goals

**Goals:**
- Make the intended dual-GPU runtime visible in the repository's launch scripts.
- Keep the change focused on practical execution changes, not distributed-training architecture.
- Preserve the existing script taxonomy while changing the GPU exposure policy.
- Clarify whether preprocessing stays single-GPU or also moves to `0,1`.
- Make dual-GPU expectations discoverable in documentation.

**Non-Goals:**
- Introduce `torchrun`, DDP, or a full distributed-training refactor unless the existing code proves it is required.
- Optimize throughput across arbitrary GPU counts.
- Generalize the repo into a cluster scheduler integration.
- Change model hyperparameters solely because the hardware is larger.

## Decisions

1. Treat `CUDA_VISIBLE_DEVICES=0,1` as the default experiment execution baseline.

Rationale:
- The user provided a specific hardware target.
- Existing scripts already encode GPU visibility directly, so the smallest coherent change is to normalize those script headers.
- The codebase already uses `device_map="auto"` in model loading, which benefits from exposing both GPUs even without a distributed-training rewrite.

Alternative considered:
- Leave scripts single-GPU and require manual shell overrides.
  Rejected because it preserves the current mismatch between repo defaults and intended hardware.

2. Keep preprocessing separate from model execution.

Rationale:
- Dataset preprocessing scripts currently run one dataset at a time and do not obviously benefit from the same execution policy as large-model training/inference.
- The proposal should avoid over-committing preprocessing to dual-GPU unless inspection during implementation shows a clear benefit or requirement.

Alternative considered:
- Force every script, including preprocessing, onto both GPUs.
  Rejected because the memory pressure and execution pattern are different.

3. Prefer script and documentation normalization first; only add runtime GPU validation if it materially improves failure messages.

Rationale:
- The main issue is launch consistency.
- A lightweight check for visible GPU count may be useful, but the repo should not be turned into a distributed launcher framework for this change.

## Risks / Trade-offs

- [Risk] Exposing two GPUs via `device_map="auto"` does not guarantee the same behavior as true DDP or tensor parallel training.
  Mitigation: scope the change to the repository's current loading strategy and document that it is a dual-visible-device setup, not a distributed rewrite.

- [Risk] Some scripts may become less convenient on single-GPU machines after normalization.
  Mitigation: document the target hardware explicitly and keep overrides possible by editing `CUDA_VISIBLE_DEVICES` at launch time.

- [Risk] Preprocessing may remain single-GPU while train/inference become dual-GPU, which is operationally asymmetric.
  Mitigation: call that out deliberately instead of pretending one policy fits all script types.

## Migration Plan

1. Identify all train/inference scripts that should expose `GPU0,GPU1`.
2. Update those scripts to use `CUDA_VISIBLE_DEVICES=0,1`.
3. Decide whether preprocessing remains `GPU0` only or moves to `GPU0,GPU1`.
4. Update documentation to state the A800 80G dual-GPU baseline.
5. Validate the resulting launch commands syntactically and, where possible, with lightweight startup checks.

## Open Questions

- Should inference scripts also default to `0,1`, or should only training scripts use both GPUs while inference remains single-GPU unless needed?
- Should the repository add an explicit startup warning when fewer than two GPUs are visible for scripts intended for the A800 dual-GPU environment?
