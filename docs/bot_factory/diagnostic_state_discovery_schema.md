# Diagnostic State Discovery Schema

Status: implemented as local diagnostic-only artifacts.

Updated: 2026-05-30 JST.

## Purpose

`diagnostic_state_discovery_v1` provides offline state clustering,
nearest-analog search, state embedding rows, suitability scoring rows, and
OOD/uncertainty calibration without allowing ML-like outputs to control
selection, readiness, or promotion.

The artifact is diagnostic research only:

```text
diagnostic_only = true
manual_review_only = true
selector_candidate_creation_allowed = false
paper_readiness_input_allowed = false
promotion_authorized_by_this_artifact = false
```

## Artifact Paths

Default writer output:

```text
data/diagnostic_state_discovery/<run_id>/diagnostic_state_discovery.json
data/diagnostic_state_discovery/<run_id>/diagnostic_state_discovery_report.md
```

CLI:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_build_diagnostic_state_discovery.py --market-state-snapshot-json data\market_state\<run_id>\market_state_snapshot.json
```

## Inputs

Required:

- one or more local `market_state_snapshot_v1` JSON artifacts.

Optional:

- local `state_conditioned_scorecard_v1` artifacts;
- local `strategy_suitability_matrix_v1` artifacts.

All input timestamps are treated as local historical as-of timestamps. Any
snapshot row with `future_data_used=true`, a feature cutoff after `data_asof`,
or a label cutoff after `data_asof` fails validation.

## Required Sections

Top-level fields:

```text
factory
schema_version
run_id
generated_at
status
input_validation
predeclared_feature_names[]
state_embedding_dataset[]
diagnostic_state_clusters[]
analog_window_search[]
ood_uncertainty_calibration
suitability_scoring_dataset[]
deterministic_label_comparison
diagnostic_model_schemas
out_of_sample_selector_replay_evidence
diagnostic_gate
safety_scope
```

`diagnostic_model_schemas` defines the local diagnostic-only schema boundaries
for `state_encoder_model_v1` and `strategy_suitability_model_v1`. Both schemas
set selector creation, paper-readiness input, and promotion authority to false.

## State Embedding Dataset

Each row records the deterministic market-state identity and a numeric feature
vector flattened from snapshot-level fields and horizon `state_vector` fields.

Rows are not selector evidence. They carry:

```text
diagnostic_only = true
selector_candidate_creation_allowed = false
paper_readiness_input_allowed = false
```

## Diagnostic State Clusters

Clusters are created from predeclared numeric features and report:

- member count;
- member timestamps;
- dominant deterministic label;
- deterministic label purity;
- centroid features;
- temporal stability placeholder;
- `INSUFFICIENT_EVIDENCE` when the cluster has too few analog windows.

Clusters cannot create selector candidates.

## Analog Window Search

Each query window searches only prior `data_asof` windows. Future windows are
not eligible analogs. Rows with too few prior analogs receive
`INSUFFICIENT_EVIDENCE`.

## OOD / Uncertainty Calibration

Calibration uses nearest historical analog distance and a local p90 threshold.
Rows are marked:

- `in_distribution`;
- `out_of_distribution`;
- `insufficient_analogs`.

The calibrated uncertainty is diagnostic-only and cannot override strict
state-conditioned evidence.

## Suitability Scoring Dataset

Suitability rows are training/evaluation examples derived from existing
state-conditioned scorecards or strategy suitability matrices. They preserve
candidate identity, state ID, horizon profile, and outcome targets, but every
row is marked diagnostic-only.

## Out-Of-Sample Boundary

The artifact always includes:

```json
{
  "beats_deterministic_baselines": false,
  "status": "not_proven",
  "selector_candidate_creation_allowed": false
}
```

Until a separate historical as-of selector replay proves out-of-sample
improvement over deterministic baselines, ML diagnostics remain advisory.
