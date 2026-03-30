# dual-gpu-execution Specification

## Purpose
TBD - created by archiving change enable-dual-gpu-a800. Update Purpose after archive.
## Requirements
### Requirement: Experiment scripts SHALL define the intended dual-GPU baseline

The repository SHALL define which experiment scripts are expected to run with `GPU0` and `GPU1` visible on the target A800 80G machine.

#### Scenario: Running a dual-GPU experiment script
- **WHEN** a user runs a training or inference script that is part of the dual-GPU baseline
- **THEN** the script SHALL expose `CUDA_VISIBLE_DEVICES=0,1`

### Requirement: The dual-GPU baseline SHALL be documented as a hardware-specific execution mode

The repository SHALL document that this execution mode targets two A800 80G GPUs rather than an abstract multi-GPU setup.

#### Scenario: Reading run instructions
- **WHEN** a user reads the repository's execution guidance
- **THEN** the guidance SHALL state that the intended hardware baseline is dual A800 80G using `GPU0` and `GPU1`

### Requirement: The repository SHALL distinguish script classes when applying the dual-GPU policy

The repository SHALL not assume that preprocessing, training, and inference all have identical GPU visibility requirements without stating that choice.

#### Scenario: Reviewing launch scripts
- **WHEN** a user compares preprocessing scripts with training or inference scripts
- **THEN** the repository SHALL make clear which script classes use the dual-GPU baseline and which do not

### Requirement: Dual-GPU launch behavior SHALL fail clearly when assumptions are violated

The repository SHALL provide either script-level or runtime-level guidance when a script intended for the dual-GPU baseline is launched without the expected device visibility.

#### Scenario: Launching with the wrong visible GPU set
- **WHEN** a user starts a dual-GPU-targeted command with fewer than the expected visible GPUs
- **THEN** the repository SHALL surface a clear indication that the launch does not match the intended dual-GPU baseline

