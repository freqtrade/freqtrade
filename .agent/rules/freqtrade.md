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
      - logic_is_proven_correct
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
      - logging_verified

# === DOCUMENTATION DUTY ===

  docs_update_required:
    on_any_trade_bot_reuse:
      - record_source_file
      - record_review_findings
      - record_changes_made
      - record_tests_run

# --- OWNERSHIP GUARD (ADD) ---

  ownership_guard:
    owner_registry:
      required: true
      file: "docs/OWNERSHIP_REGISTRY.yaml"
    checks_required:
      - detect_duplicate_class_definitions
      - detect_reexports_and_alias_exports
      - detect_multi_instantiated_singletons
      - verify_imports_target_owner_module
    handling: STOP_ON_FIRST_VIOLATION

# --- SEQUENCING GUARD (ADD) ---

  sequencing_guard:
    objective: "Ensure runtime lifecycle order is explicit, deterministic, and testable."
    lifecycle_spec:
      required: true
      file: "docs/LIFECYCLE_SPEC.yaml"
      must_define:
        - states
        - transitions
        - startup_order
        - shutdown_order
        - invariants
    checks_required:
      import_side_effect_scan:
        forbid:
          - "network_calls_on_import"
          - "scheduler_start_on_import"
          - "thread_start_on_import"
        commands:
          - "rg -n \"^(?!\\s*#).*\\b(requests\\.|httpx\\.|aiohttp\\.|websocket\\.|BreezeConnect\\(|create_task\\(|Thread\\(|schedule\\.|patch_ccxt\\()\" src/ adapters/ freqtrade/ tests/ || true"
      orchestrator_scan:
        rule: "Only main/orchestrator may call bootstrap/start loop."
        commands:
          - "rg -n \"\\b(start|run|bootstrap|init)_(scheduler|loop|engine)\\b\" src/ adapters/ | head -n 200"
      tracer_gate:
        requirement: "viztracer trace must show expected sequence markers"
        markers_required:
          - "BOOT:config"
          - "BOOT:logging"
          - "BOOT:rate_limiter"
          - "BOOT:exchange_adapter"
          - "BOOT:strategy_loop"
          - "SHUTDOWN:stop_loop"
          - "SHUTDOWN:cleanup"
    violation_handling:
      - STOP_ON_FIRST_VIOLATION
      - require_fix_before_any_new_work

# =========================

# ADD: TIME DETERMINISM WORKSPACE CHECKS

# =========================

  time_determinism_guard:
    objective: "Prevent flaky gates by banning wall-clock and sleep in unit tests and requiring injectables."
    checks_required:
      ban_wall_clock_in_tests:
        commands:
          - "rg -n \"\\b(time\\.sleep\\(|asyncio\\.sleep\\(|datetime\\.now\\(|time\\.time\\(|perf_counter\\()\" tests/ || true"
      require_injection_for_time_deps:
        commands:
          - "rg -n \"class\\s+(RateLimiter|AlertManager)\\b|def\\s+__init__\\(.*(now_fn|sleep_fn)\" adapters/ccxt_shim/ || true"
    violation_handling:
      - STOP_ON_FIRST_VIOLATION

# =========================

# ADD: EXIT SELL PATH WORKSPACE CHECKS

# =========================

  exit_policy_guard:
    objective: "Ensure exits are possible in real mode without permitting shorts."
    checks_required:
      detect_sell_block_without_escape_hatch:
        commands:
          - "rg -n \"buyer_only|assert_buyer_only|Sell orders are disabled\" adapters/ccxt_shim/ | head -n 200"
      require_reduceonly_or_position_source:
        commands:
          - "rg -n \"reduceOnly|ft_exit|fetch_positions\\(\" adapters/ccxt_shim/ | head -n 200"
      require_test_for_exit_path:
        commands:
          - "rg -n \"reduceOnly|ft_exit\" tests/ | head -n 200"
    violation_handling:
      - STOP_ON_FIRST_VIOLATION
