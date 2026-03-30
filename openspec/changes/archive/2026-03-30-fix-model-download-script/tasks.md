## 1. Audit The Download Contract

- [x] 1.1 Compare every `download_models.sh` target with the local paths used by runtime model resolution and auxiliary model-loading code.
- [x] 1.2 Identify whether any current path mismatch is a real bug, an intentional alias, or missing documentation.

## 2. Fix The Script If Needed

- [x] 2.1 Correct any incorrect model download targets, directory names, or comments discovered during the audit.
- [x] 2.2 Add minimal script hardening or structure improvements if they help setup fail clearly and early.

## 3. Verify And Record Expectations

- [x] 3.1 Re-check that the final download layout matches the runtime contract for supported model keys.
- [x] 3.2 Update any user-facing guidance that would otherwise leave the expected model download layout ambiguous.
