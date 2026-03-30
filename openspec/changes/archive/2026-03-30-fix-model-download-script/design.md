## Context

The repository uses local model directories as a contract between setup scripts and runtime code.

Current relevant pieces:
- `download_models.sh` downloads seven model assets into `./models/...`
- `src/model/__init__.py` resolves runtime keys such as `7b`, `7b_chat`, `table_llama_7b`, `mistral_7b`, and `mistral_7b_instruct`
- other code paths also expect local auxiliary assets such as:
  - `models/google-bert/bert-base-uncased`
  - `models/sentence-transformers/all-roberta-large-v1`

The key risk is not only "wrong repo name". It is the broader setup contract:

```text
download_models.sh
      │
      ▼
local models/... directories
      │
      ▼
runtime path resolution in src/model/__init__.py
      │
      ├── train / inference entrypoints
      └── smoke tests / auxiliary model loads
```

If any path in that chain drifts, the user gets a local asset failure during startup.

## Goals / Non-Goals

**Goals:**
- Verify that every runtime-referenced downloaded model path is either created directly by the script or intentionally handled via aliases.
- Fix any discovered mismatch in a single place, with a preference for minimal and clear setup behavior.
- Make the download script more robust and self-explanatory if the current script is too bare for reliable setup.
- Preserve compatibility with the runtime path aliases already introduced where sensible.

**Non-Goals:**
- Redesign model loading across the repository.
- Add remote fallback loading during training or inference.
- Change which backbone models the repository supports.
- Solve Hugging Face authentication or gated-model access beyond documenting the requirement.

## Decisions

1. Treat the download script and runtime resolver as one contract surface.

Rationale:
- Users experience both as a single setup path.
- Fixing only comments or only runtime aliases would leave the overall setup story fragile.

2. Prefer correcting the download script to match the runtime model names when there is a clear canonical path.

Rationale:
- The setup script is where users establish local state.
- A clean download layout reduces the need for defensive runtime aliases.

Alternative considered:
- Rely only on runtime alias handling.
  Rejected because it can mask setup drift and make the expected local layout unclear.

3. Keep robustness improvements lightweight.

Rationale:
- `download_models.sh` should stay simple to run and inspect.
- Minimal shell hardening such as `set -euo pipefail` and grouped comments improves reliability without changing behavior dramatically.

## Risks / Trade-offs

- [Risk] Some users may already have models downloaded in a previously accepted layout.
  Mitigation: keep runtime aliases where they help preserve compatibility, unless they conflict with clarity.

- [Risk] Gated models such as Llama 2 may still fail to download even with a correct script.
  Mitigation: document that these failures are access/auth issues rather than path mismatches.

- [Risk] Tightening the script may expose setup problems earlier.
  Mitigation: prefer fail-fast behavior with actionable error context.

## Migration Plan

1. Compare `download_models.sh` targets with runtime model path resolution and auxiliary model loads.
2. Fix incorrect download destinations or naming mismatches if found.
3. Add lightweight shell hardening or comments if they materially improve setup reliability.
4. Verify the script statically and re-check runtime path expectations.

## Open Questions

- Should the script normalize the `7b_chat` local directory to the exact canonical runtime alias, or is the current runtime alias support sufficient?
- Should auxiliary downloads such as sentence-transformer and BERT be documented separately from the primary LLM backbones?
