## 1. Define The Dual-GPU Policy

- [x] 1.1 Identify which experiment scripts should default to `CUDA_VISIBLE_DEVICES=0,1` on the target A800 80G machine.
- [x] 1.2 Decide whether preprocessing remains single-GPU or should also expose both GPUs.

## 2. Normalize Launch Behavior

- [x] 2.1 Update the selected training and inference scripts to use the dual-GPU baseline consistently.
- [x] 2.2 Add any minimal runtime or script-level validation needed for clearer failures when the expected GPU visibility is not met.

## 3. Record The Hardware Baseline

- [x] 3.1 Update repository-facing instructions to state that the intended execution environment is dual A800 80G on `GPU0` and `GPU1`.
- [x] 3.2 Verify the updated commands and record any remaining limitations, especially around `device_map=\"auto\"` versus true distributed training.
