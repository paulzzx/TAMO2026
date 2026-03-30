## ADDED Requirements

### Requirement: Llama 2 experiment scripts SHALL target Llama 2 model variants

Experiment scripts named for the Llama 2 family SHALL not silently launch Mistral-based prompt or model variants.

#### Scenario: Running a Llama 2 training script
- **WHEN** a user runs a `script/llama2_*.sh` training script
- **THEN** the script SHALL pass a Llama 2-compatible `prompt_type`
- **AND** the script SHALL select a Llama 2-compatible `model_name`
- **AND** the script SHALL select a Llama 2-compatible `llm_model_name`

#### Scenario: Running a Llama 2 inference script
- **WHEN** a user runs `script/llama2_inference.sh`
- **THEN** the script SHALL not route inference through Mistral prompt/model selections

### Requirement: Runtime configuration SHALL use valid registered model keys

The Python experiment entrypoints SHALL not depend on impossible default model selections.

#### Scenario: Starting an entrypoint with defaults
- **WHEN** a user invokes `table_train.py` or `inference.py` without overriding the default `model_name`
- **THEN** the selected default SHALL exist in `src.model.load_model`

#### Scenario: Starting an entrypoint with explicit model keys
- **WHEN** a user supplies an unknown `model_name` or `llm_model_name`
- **THEN** the startup path SHALL fail fast with a clear validation error

### Requirement: Model-path resolution SHALL match supported local asset layouts

The runtime SHALL resolve local model paths in a way that is consistent with the repository's download scripts and supported historical directory layouts.

#### Scenario: Using downloaded local checkpoints
- **WHEN** a user downloads model assets with the repository's download script
- **THEN** the runtime SHALL resolve those local directories successfully for the corresponding model keys

#### Scenario: A required model directory is missing
- **WHEN** the selected model key resolves to no supported local directory
- **THEN** the startup path SHALL fail with a clear error that identifies the missing asset

### Requirement: Reproduction-facing references SHALL not point to nonexistent scripts

User-facing reproduction instructions SHALL not direct users to paths that do not exist in the repository.

#### Scenario: Following the documented preprocess workflow
- **WHEN** a user follows the preprocess instruction from the repository documentation
- **THEN** the referenced script path SHALL exist in the repository
