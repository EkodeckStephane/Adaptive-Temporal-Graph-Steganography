# Research artefact — Adaptive Graph Traversal Steganography

This directory contains the Python package, scripts, configurations and tests for the paper *Adaptive Distribution-Preserving Steganography through Learned Traversals of Temporal Graphs*.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick validation

```bash
pytest
python scripts/validate_project.py
python scripts/validate_research_spec.py
```

## Reproducing the main results

The experiments are organised by phase. Each script is deterministic when the
project seed is respected.

### Cover models (Phase 4)

```bash
python scripts/run_phase4_temporal_cover_model.py
python scripts/run_phase4_neural_cover_model.py
python scripts/run_phase4_spatial_sensitivity.py
```

### Codec and controller contracts (Phases 5–6)

```bash
python scripts/run_phase5_codec.py
python scripts/run_phase6_fuzzy_controller.py
```

### External steganalysis and robustness (Phases 7–8)

```bash
python scripts/run_phase7_steganalysis.py
python scripts/run_phase8_robustness.py
```

### Closing experiments (Phase 9)

```bash
python scripts/run_phase9_temporal_gnn_cover_model.py
python scripts/run_phase9_independent_neural_steganalysis.py
python scripts/run_phase9_active_channel_reliability.py
```

Results are written to `../results/tables/` and `../results/figures/`.

## Package structure

- `src/data/` — dataset adapters, causal splits, spatial discretization.
- `src/models/` — cover models (temporal backoff, neural sequence, TGN-style).
- `src/stego/` — encryption, coding, active-channel reliability.
- `src/controllers/` — fuzzy controller and baselines.
- `src/steganalysis/` — detectors, neural Eves, robustness tests.
- `src/baselines/` — Lee BIND/AdaBIND/BYNIS reproduction and walk baselines.
- `configs/` — master configuration and per-phase YAMLs.
- `scripts/` — deterministic entry points for each experimental phase.
- `tests/` — property tests and integration tests.

## Current implementation choices

The codec now offers two distribution-coupling backends:
- **quantized-interval** (legacy): uniform dyadic intervals over the top candidates.
- **range** (improved): dyadic intervals proportional to the cover-model
  probabilities, yielding lower KL divergence and total variation.

Two inner codes are available:
- **repetition-3**: simple majority-vote repetition.
- **reed_solomon**: algebraic block code via `reedsolo`, correcting several bit
  flips per block.

The fuzzy controller exposes its Takagi–Sugeno consequence weights as tunable
parameters. Use `scripts/optimize_fuzzy_weights.py` to fit them on a validation
objective that trades payload against risk and abstention.

The temporal graph cover model includes a node-memory GRU, sinusoidal time
encoding, and a GAT-style attention readout, with early stopping and
post-hoc temperature scaling for calibration.

## Datasets and licences

Raw third-party datasets are not redistributed. Preprocessing scripts,
checksums and provenance notes are provided so that each dataset can be
re-acquired from its original source. See `../docs/data/` and
`../docs/decisions/architecture_decision_003.md` for details.
