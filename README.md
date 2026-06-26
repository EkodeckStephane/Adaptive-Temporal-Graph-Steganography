# Adaptive Temporal Graph Steganography

This repository contains the reproducible software artifact and derived
results for adaptive distribution-preserving steganography through learned
traversals of temporal graphs.

## Contents

- `code/`: Python package, phase scripts, configurations, and tests.
- `experiments/`: frozen YAML experiment definitions.
- `results/`: derived CSV/JSON result tables.
- `datasets/metadata/`: dataset metadata and reacquisition notes.
- `literature/external/lee_hmg_snapshot/`: minimal external code snapshot used
  by the HMG/BIND reproduction tests.
Raw third-party datasets are not redistributed here. The `datasets/raw`,
`datasets/interim`, and `datasets/processed` directories contain placeholders
only; use the metadata and preprocessing scripts to reacquire and regenerate
the derived data under the original dataset licenses.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\requirements.txt
```

## Validation

```powershell
python -m ruff check code\src code\tests code\scripts
$env:PYTHONPATH=(Resolve-Path 'code\src').Path; python -m pytest
python code\scripts\validate_project.py
python code\scripts\validate_research_spec.py
```

## Main Experiment Commands

```powershell
python code\scripts\run_phase7_steganalysis.py
python code\scripts\run_phase8_robustness.py
python code\scripts\run_phase9_temporal_gnn_cover_model.py
python code\scripts\run_phase9_independent_neural_steganalysis.py
python code\scripts\run_phase9_adaptive_steganalysis.py
python code\scripts\run_phase10_adaptive_hardening_sweep.py
python code\scripts\run_phase9_active_channel_reliability.py
```

The derived result tables used for reporting are in `results/tables`.
