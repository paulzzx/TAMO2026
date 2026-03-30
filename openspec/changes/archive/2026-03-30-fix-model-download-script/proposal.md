## Why

`download_models.sh` is a critical setup step for TAMO, because runtime model resolution depends on the exact local directory layout it creates. The current script appears broadly aligned with the codebase, but it still deserves a focused audit because a mismatch here produces expensive setup failures later, after dependencies and datasets are already in place.

This change proposes a targeted check-and-fix pass for the model download script so that the downloaded assets, local paths, and user guidance remain consistent with the repository's current runtime expectations.

## What Changes

- Audit `download_models.sh` against the model keys and local directory paths used at runtime.
- Fix any incorrect download targets, path mismatches, or naming inconsistencies discovered during the audit.
- Improve script robustness where needed, such as fail-fast behavior and clearer user-facing comments or grouping.
- Record the expected correspondence between downloaded model directories and runtime model names.

## Capabilities

### New Capabilities
- `model-download-consistency`: Define the contract between the model download script and the runtime code that resolves local model assets.

### Modified Capabilities
- None.

## Impact

Affected areas:
- `download_models.sh`
- `src/model/__init__.py`
- any documentation that references the model download flow
- smoke-test or runtime guidance that depends on downloaded asset paths

This change is limited to setup-time model asset handling and related documentation.
