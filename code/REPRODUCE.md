# Reproduction instructions for Q1 submission

This document lists the exact commands needed to reproduce the main tables and
figures of the paper. All commands are deterministic when the project seed is
respected.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Validation

```bash
pytest -q --basetemp=./.pytest_tmp
python scripts/validate_project.py
python scripts/validate_research_spec.py
```

## Phase 4 — Cover models and spatial sensitivity

```bash
python scripts/run_phase4_temporal_cover_model.py
python scripts/run_phase4_neural_cover_model.py
python scripts/run_phase4_spatial_sensitivity.py
```

## Phase 5 — Codec contract

```bash
python scripts/run_phase5_codec.py
```

## Phase 6 — Fuzzy controller and weight optimization

```bash
python scripts/run_phase6_fuzzy_controller.py
python scripts/optimize_fuzzy_weights.py
```

## Phase 7 — External steganalysis

```bash
python scripts/run_phase7_steganalysis.py
```

## Phase 8 — Robustness and ablations

```bash
python scripts/run_phase8_robustness.py
```

## Phase 9 — Closing experiments

```bash
python scripts/run_phase9_temporal_gnn_cover_model.py
python scripts/run_phase9_independent_neural_steganalysis.py
python scripts/run_phase9_adaptive_steganalysis.py
python scripts/run_phase9_active_channel_reliability.py
```

## Outputs

All JSON/CSV outputs are written to `../results/tables/` and figures to
`../results/figures/`. The final article and thesis PDFs are compiled from
`../article/main.tex` and `../thesis/main.tex`.

## Notes

- The improved codec backend (`range`) and Reed--Solomon error correction are
  implemented in `src/stego/coding.py`. The legacy backend remains available for
  ablation.
- The fuzzy controller weights are optimized by `scripts/optimize_fuzzy_weights.py`.
- The adaptive steganalyst (`scripts/run_phase9_adaptive_steganalysis.py`) uses
  internal codec features and therefore represents a stronger adversary than the
  independent neural Eves.
