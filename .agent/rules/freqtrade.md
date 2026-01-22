---
trigger: always_on
---

AG_WORKSPACE:

# === REUSE FROM trade-bot REPO (HARD GUARDRAILS) ===

reuse_policy_trade_bot:
  trust_level: "LOW"
  presumption: "INCORRECT_UNTIL_PROVEN"
  allow_reuse: true
  but_require:
    - mandatory_review_before_copy
    - mandatory_functional_validation
    - mandatory_architecture_alignment_check

  review_mandate:
    when_triggered:
      - any_file_copied_from_trade_bot
      - any_function_reused_from_trade_bot
      - any_pattern_inspired_by_trade_bot
    review_steps_must_cover:
      - behavior_correctness_check
      - edge_case_analysis
      - rate_limit_safety
      - error_handling_safety
      - security_sanitization
      - dependency_direction_compliance
      - testability_assessment

  acceptance_criteria_before_merge:
    - logic_is_proven_correct (via tests or tracing)
    - behavior_matches_freqtrade_and_breeze_docs
    - no_hidden_side_effects
    - no_leaked_secrets
    - no_layer_violation

# === PORTS & ADAPTERS RULE (CLEAN ARCHITECTURE) ===

ports_and_adapters:
  principle: "DEPENDENCY_INWARD_ONLY"
  rules:
    - outer_layers_may_depend_on_inner_layers
    - inner_layers_must_never_depend_on_outer_layers
    - infrastructure_must_depend_on_application_via_ports
    - application_must_depend_on_domain_via_interfaces
    - domain_must_have_zero_dependencies_on_infrastructure

  ports_must_be:
    - explicit_interfaces
    - minimal
    - stable
    - documented

  forbidden:
    - cross_layer_imports
    - presentation_importing_application_internals
    - strategy_importing_broker_or_sdk
    - domain_importing_streamlit_or_any_ui
    - application_importing_infrastructure_details

# === DEPENDENCY DIRECTION ENFORCEMENT ===

dependency_guard:
  checks_required:
    - pydeps_direction_validation
    - import_linter_layer_check
    - viztracer_lifecycle_check

  violations_handling:
    - STOP_ON_FIRST_VIOLATION
    - require_fix_before_any_new_work

# === FUNCTIONALITY VALIDATION WHEN REUSING CODE ===

functional_validation:
  required_for_any_trade_bot_reuse:
    - unit_tests_exist_or_added
    - integration_test_with_mock_breeze
    - happy_path_verified
    - failure_path_verified
    - rate_limit_behavior_verified
    - logging_verified (no secrets)

# === DOCUMENTATION DUTY ===

docs_update_required:
  on_any_trade_bot_reuse:
    - record_source_file
    - record_review_findings
    - record_changes_made
    - record_tests_run
