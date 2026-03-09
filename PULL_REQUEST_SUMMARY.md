# Pull Request Summary: MTF Regime Detection + Phase 1 Calibration Tooling

## Overview

This PR adds multi-timeframe regime detection, score-band calibration/reporting, and a complete Phase 1 test harness for reproducible GA comparisons.

It introduces:
- continuous regime scoring upgrades (`advanced_ensemble` + score-band segmentation),
- MTF fusion (`hierarchical` and `weighted_voting`) with transition/context signals,
- new island/MTF/Phase1 configs and runners,
- diagnostics and calibration tooling,
- dedicated regression tests for the new behavior.

---

## Scope Included

### Core logic
- `genetic_algorithm/utils/regime_detector.py`
  - Adds `advanced_ensemble` detection path.
  - Adds continuous score APIs and score-band segmentation.
  - Improves segment splitting with scarcity-aware holdout behavior.
- `genetic_algorithm/utils/mtf_regime_detector.py`
  - New module for MTF score fusion, transition detection, and context labeling.
- `genetic_algorithm/utils/dataset_policy.py`
  - Adds MTF-aware auto-holdout path and segmentation mode selection.

### GA/strategy/ML integration
- Updated integration points:
  - `genetic_algorithm/core/island_model.py`
  - `genetic_algorithm/core/mutation.py`
  - `genetic_algorithm/core/strategy_gene.py`
  - `genetic_algorithm/strategies/generator.py`
  - `genetic_algorithm/ml/regime_detector.py`
  - `genetic_algorithm/ml/regime_trainer.py`
  - `genetic_algorithm/ml/train_regime.py`

### Tooling and test harness
- New tools:
  - `genetic_algorithm/tools/calibrate_bands.py`
  - `genetic_algorithm/tools/phase1_diagnostics.py`
- New runners:
  - `genetic_algorithm/scripts/run_phase1_tests.py`
  - `genetic_algorithm/scripts/run_island_v2_tests.sh`
  - `run_island_mtf_tests.sh`
- New tests:
  - `genetic_algorithm/tests/test_mtf_regime.py`

### Config additions
- Added MTF/island and Phase 1 config matrix under `genetic_algorithm/config/`:
  - `ga_config_island_mtf_*.yaml`
  - `ga_config_island_v2_*.yaml`
  - `ga_config_mtf_test*.yaml`
  - `ga_config_phase1_*.yaml`

---

## PR Hygiene / Cleanup

- Verified changed files are source/config/test/tooling only.
- Verified no generated `output` artifacts or local strategy outputs are included in the diff.
- Existing ignore rules already cover:
  - `genetic_algorithm/output/`
  - `genetic_algorithm/logs/`
  - `user_data/*` (including strategy artifacts)

---

## Validation Status

- Editor/static diagnostics: no errors reported in modified files.
- Runtime tests: local environment missing Python dependencies (`numpy`) and pytest plugin args from project config required override, so full test run was not completed in this environment.

---

## Suggested Next Improvements

1. **Stabilize CI/dev test environment**
   - Ensure required deps/plugins (`numpy`, pytest-xdist, etc.) are installed by default to avoid false-negative local validation.

2. **Tighten calibration quality gates**
   - Enforce minimum regime coverage/segment thresholds as hard fail in CI for new configs to prevent poor splits from passing.

3. **Unify config matrix lifecycle**
   - Add a compact index/doc for the new config families (MTF v1/v2/Phase1) and expected use-cases to reduce maintenance drift.

4. **Add benchmark snapshots to PR checks**
   - Capture quick smoke metrics (runtime, best fitness, regime coverage) for A/B/C/D runs to make regressions visible earlier.

5. **Harden MTF fallback behavior observability**
   - Emit explicit counters in diagnostics for fallback paths (single-TF fallback, scarce-regime holdout waiver) to simplify production troubleshooting.
