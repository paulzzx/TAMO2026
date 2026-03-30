# model-download-consistency Specification

## Purpose
TBD - created by archiving change fix-model-download-script. Update Purpose after archive.
## Requirements
### Requirement: The model download script SHALL match runtime model asset expectations

The repository SHALL keep the model download script aligned with the local paths expected by runtime model resolution.

#### Scenario: Downloading supported model assets
- **WHEN** a user runs `download_models.sh`
- **THEN** the downloaded local directories SHALL match the supported runtime model keys directly or through explicitly supported aliases

### Requirement: Auxiliary downloaded model assets SHALL remain discoverable by the code that uses them

The repository SHALL keep non-backbone downloaded model assets aligned with the code paths that load them.

#### Scenario: Loading auxiliary local assets
- **WHEN** code loads local assets such as BERT or sentence-transformer checkpoints from `models/...`
- **THEN** the repository SHALL provide a documented or directly created download path for those assets

### Requirement: Download-script failures SHALL be easier to interpret than downstream runtime path failures

The repository SHALL prefer setup-time clarity over delayed path mismatch failures during experiment startup.

#### Scenario: A user runs the model download flow
- **WHEN** the script encounters a setup problem that prevents required model assets from being prepared correctly
- **THEN** the script or associated guidance SHALL make the failure easier to diagnose than a later missing-path error during training or inference

