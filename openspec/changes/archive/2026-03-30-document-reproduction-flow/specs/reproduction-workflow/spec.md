## ADDED Requirements

### Requirement: The project SHALL define a minimum local validation workflow

The repository SHALL document a lightweight path for validating that the project structure and core non-server code paths are intact without requiring full Linux/CUDA experiment setup.

#### Scenario: Local validation workflow is followed
- **WHEN** a contributor wants to validate the repository on a local development machine
- **THEN** the documented workflow SHALL point them to the local dependency set
- **AND** the workflow SHALL identify `smoke_test_local.py` as the validation command
- **AND** the workflow SHALL state that this path validates dataset loading, graph collation, and a mocked pure-text model path rather than paper-level training

### Requirement: The project SHALL define a full experiment reproduction workflow

The repository SHALL document the ordered prerequisites for running TAMO experiments in the intended server environment.

#### Scenario: Full experiment reproduction is attempted
- **WHEN** a contributor wants to reproduce TAMO training or inference results
- **THEN** the workflow SHALL require environment setup before any experiment script is run
- **AND** the workflow SHALL require local model assets and local benchmark datasets to exist before training or inference
- **AND** the workflow SHALL require preprocessing to generate graph artifacts before any hypergraph model variant is executed
- **AND** the workflow SHALL identify the train and inference entrypoints used by the shell scripts

### Requirement: The workflow SHALL distinguish pure-text and TAMO hypergraph runs

The repository SHALL not present all experiments as if they had the same prerequisites.

#### Scenario: A user chooses an experiment family
- **WHEN** the user selects a pure-text model variant
- **THEN** the workflow SHALL make clear that graph preprocessing artifacts are not the primary dependency
- **WHEN** the user selects a TAMO hypergraph variant
- **THEN** the workflow SHALL make clear that `graphs/*.pt` artifacts are required inputs

### Requirement: The workflow SHALL identify verification points and output locations

The documented flow SHALL define what counts as success after each stage and where results are written.

#### Scenario: A run finishes
- **WHEN** training or inference completes
- **THEN** the workflow SHALL identify `output_dir/<dataset>/score.txt` as a result checkpoint
- **AND** the workflow SHALL explain that evaluation files are also written under `output_dir/<dataset>/`

### Requirement: The workflow SHALL record known repository mismatches that affect reproduction

The documented flow SHALL surface important mismatches instead of assuming the scripts and README are perfectly aligned.

#### Scenario: A contributor follows the recorded workflow
- **WHEN** the repository contains a path mismatch, naming inconsistency, or script-label mismatch relevant to reproduction
- **THEN** the workflow SHALL call it out explicitly as a risk or caveat
