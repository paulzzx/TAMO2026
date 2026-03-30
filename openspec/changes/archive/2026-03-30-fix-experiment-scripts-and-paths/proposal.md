## Why

Several experiment entry scripts and runtime defaults are inconsistent with the repository's actual model registry and downloaded asset layout. These mismatches make the experiment suite brittle: some `llama2` scripts actually launch Mistral variants, and some runs can fail immediately due to model-path or default-key mismatches.

This change proposes a focused hardening pass so the published experiment scripts match their intended model families and common reproduction-time failures are handled in the codebase instead of being left to manual debugging.

## What Changes

- Correct `llama2` experiment scripts that currently point at Mistral prompts or checkpoints.
- Align model path resolution with the directories created by `download_models.sh`.
- Fix obvious runtime-default mismatches that can cause entrypoints to fail before training or inference starts.
- Add light validation or fail-fast checks around model selection and required local assets.
- Update reproduction-facing documentation/scripts where they encode incorrect paths or names.

## Capabilities

### New Capabilities
- `experiment-script-consistency`: Ensure experiment launch scripts and runtime configuration consistently target the intended model family and asset layout.

### Modified Capabilities
- None.

## Impact

Affected areas:
- `script/llama2_*.sh`
- `script/tablellama_*.sh`
- `script/mistral_*.sh`
- `src/model/__init__.py`
- `src/config.py`
- `table_train.py`
- `inference.py`
- `README.md`
- model download and smoke-test paths

This change is limited to experiment orchestration, path handling, and startup validation. It does not redesign model architectures or dataset preprocessing logic.
