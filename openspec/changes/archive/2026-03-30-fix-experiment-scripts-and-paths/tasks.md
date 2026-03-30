## 1. Script Consistency

- [x] 1.1 Correct the `llama2` experiment scripts so their `prompt_type`, `model_name`, and `llm_model_name` target Llama 2 rather than Mistral.
- [x] 1.2 Review the remaining experiment scripts for family/path inconsistencies and fix any directly related launch-time mismatches.

## 2. Runtime Validation

- [x] 2.1 Fix invalid runtime defaults such as the nonexistent default `model_name` in `src/config.py`.
- [x] 2.2 Harden model-path resolution in `src/model/__init__.py` so it matches the directories produced by `download_models.sh` or explicitly supports both known layouts.
- [x] 2.3 Add fail-fast checks in the train/inference startup path for unknown model keys and missing local model assets.

## 3. Reproduction Hardening

- [x] 3.1 Fix incorrect reproduction-facing references such as the README preprocess script path.
- [x] 3.2 Verify the corrected flow with smoke tests or equivalent startup validation and record any remaining limitations.
