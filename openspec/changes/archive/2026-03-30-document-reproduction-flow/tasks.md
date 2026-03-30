## 1. Establish The Reproduction Baseline

- [ ] 1.1 Confirm the two supported paths: local smoke validation and Linux/CUDA experiment reproduction.
- [ ] 1.2 Record the required environment inputs for each path, including Python version, PyTorch/PyG expectations, dataset assets, and model assets.

## 2. Record The Ordered Workflow

- [ ] 2.1 Document the minimum local validation flow: install local dependencies, verify local datasets, run `smoke_test_local.py`.
- [ ] 2.2 Document the server flow: install runtime dependencies, download datasets and models, generate graph artifacts, run server smoke test, then launch train or inference scripts.
- [ ] 2.3 Record where outputs and scores are written so results can be checked after each run.

## 3. Record Reproduction Risks

- [ ] 3.1 Document repository mismatches that can break reproduction if followed literally.
- [ ] 3.2 Separate pure-text baselines from TAMO hypergraph runs so preprocessing requirements are not missed.
