## Context

The repository exposes experiments through shell scripts under `script/`, while Python entrypoints resolve model names through `src/model/__init__.py` and argument defaults from `src/config.py`.

Current inspection found several concrete inconsistencies:
- `script/llama2_pt.sh` uses `prompt_type mistral`, `pt_mistral`, `table_hypergraph_mistral`, and `mistral_7b` even though the file is labeled as a Llama 2 experiment script.
- `script/llama2_inference.sh` uses Mistral prompt/model selections despite being the Llama 2 inference script.
- `src/model/__init__.py` expects the chat checkpoint under `models/meta-llama/Llama-2-7b-chat-hf`, but `download_models.sh` downloads to `models/meta-llama/Llama-2-7b-chat`.
- `src/config.py` defaults `model_name` to `table_graph_llm`, which is not a valid key in `load_model`.
- `README.md` still points users to `data_preprocesss.sh`, a path that does not exist.

These are not isolated documentation issues. They affect runtime behavior directly:

```text
shell script
   │
   ▼
CLI args in table_train.py / inference.py
   │
   ├── model_name must exist in load_model
   ├── llm_model_name must exist in llama_model_path
   └── resolved local path must match downloaded assets
```

If any of those three contracts are broken, reproduction fails before the experiment itself is evaluated.

## Goals / Non-Goals

**Goals:**
- Make every `llama2` script actually target Llama 2 prompt/model variants.
- Make model path resolution consistent with downloaded directory names or support both known layouts.
- Remove or correct startup defaults that point to nonexistent model keys.
- Add explicit validation for invalid model selections or missing local assets.
- Repair user-facing reproduction references that are provably wrong.

**Non-Goals:**
- Rewrite the experiment taxonomy or rename every historical script.
- Change model architecture, training hyperparameters, or evaluation logic beyond what is needed to make launches consistent.
- Introduce a new experiment launcher framework.
- Solve every possible environment issue unrelated to the discovered mismatches.

## Decisions

1. Normalize `llama2` scripts to the Llama 2 family instead of relabeling them as Mistral scripts.

Rationale:
- The filenames already communicate user intent.
- The repo already has Llama 2 model keys and download commands.
- Fixing the commands is lower-cost and less disruptive than renaming the experiment surface.

Alternative considered:
- Rename the affected scripts as Mistral scripts.
  Rejected because it would preserve the broken user expectation that current `llama2` scripts are valid.

2. Centralize model-path correctness in Python instead of depending only on shell-script discipline.

Rationale:
- Shell scripts can still drift again.
- Runtime path validation can catch both manual CLI invocations and scripted runs.
- Supporting both `.../Llama-2-7b-chat` and `.../Llama-2-7b-chat-hf` is safer than assuming only one historical layout.

Alternative considered:
- Fix `download_models.sh` only.
  Rejected because existing local environments may already have the current downloaded layout.

3. Replace invalid defaults with valid, conservative defaults.

Rationale:
- Entry scripts should fail only when the user asks for something invalid, not because the built-in defaults are impossible.
- A default `model_name` that is absent from `load_model` is a guaranteed startup bug.

Alternative considered:
- Leave defaults unchanged and rely entirely on scripts.
  Rejected because the entrypoints are part of the public interface.

4. Add fail-fast validation around model keys and required local paths.

Rationale:
- Current failures are likely to show up later in the stack as less actionable exceptions.
- A short validation message at startup materially reduces debugging cost.

## Risks / Trade-offs

- [Risk] Historical runs may have been launched with the old mislabeled scripts and therefore used different backbones than their filenames imply.
  Mitigation: keep the change focused on future correctness and document the intended mapping in the change artifacts.

- [Risk] Supporting multiple local checkpoint directory names can hide environment drift if not documented.
  Mitigation: validate known aliases explicitly and keep the accepted mapping narrow.

- [Risk] Changing defaults can alter behavior for users who relied on implicit CLI defaults.
  Mitigation: choose defaults that correspond to real registered models and preserve script-level explicit arguments.

## Migration Plan

1. Correct the affected shell scripts to use consistent prompt/model combinations.
2. Update runtime model-path mapping and add startup validation helpers.
3. Fix obviously incorrect README/script references.
4. Verify with smoke tests and at least one no-op CLI parse or startup path per family.

Rollback strategy:
- Revert the script and path-validation changes together so filenames and runtime expectations remain aligned.

## Open Questions

- Should the repo prefer `7b` or `7b_chat` as the canonical Llama 2 inference default for instruction-style evaluation scripts?
- Should missing local model assets be a hard error in both train and inference entrypoints, or only when the selected model family is instantiated?
