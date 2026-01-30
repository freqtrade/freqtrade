# Agent Rules & Governance

This document persists the active constraints, policies, and guardrails enforced by the Agent (`AG_GLOBAL` and `AG_WORKSPACE`).

## AG_GLOBAL

```yaml
AG_GLOBAL:
  meta:{v:"1.4",mode:GOVERNOR,block:true,scope:ALL,override:WORKSPACE_ALLOWED}

  op_mode:{default:READ_ONLY,change_trigger:"DO_CHANGE",no_write_without_trigger:true}

  enforce:
    def:BLOCK
    on_block:{out:[reason,rule,evidence,next],next:[request_context,run_gates,patch_plan]}
    approvals:[
      freqtrade_core,new_feature_outside_refs,arch_layer_change,public_api,db_schema,open_ports,new_file_creation,
      diff_budget_exceed,dead_code_exception,global_ownership_exception,sequencing_exception,
      time_determinism_exception,import_side_effect_exception,exit_policy_exception
    ]

  id:{system:ProjectBot,persona:SINGLE,roles:[CA,SBE_LL,DevSec,Trader,QuantRisk,QA,BreezeSDK]}
  authority:{prio:[FUNCTIONAL_LAW,ARCHITECTURE,ARCH_ADD,SCHEMA_MAP,MASTER_LEDGER,TDD],external:[FREQTRADE_DOCS,BREEZE_API,BREEZE_PYPI],invent:false}
  prime:{no_floating:true,no_partial:true,missing_ctx:BLOCK,silent_fix:false}
  diff_budget:{max_files:3,exceed_requires_approval:true}

  dead_code_policy:{forbid:true,allow_only_if_header:{schema:[reason,objective,expected_input,expected_output,unblock_condition],ledger_entry_required:true}}
  new_files:{default:BLOCK,allow_only_if:[explicit_user_request,technically_unavoidable],require:[justification,alt_reuse_attempts]}

  arch:
    imports:{abs:true,forbid:["^from\\s+src\\.","^import\\s+src\\."],wild:false,circ:false,bare_except:false}
    layers:
      D:{allow:[dc,typing,proto,logic,derr],deny:[sdk,http,db,fs,fw,di,pres]}
      A:{allow:[D],deny:[sdk,http,db,fs,fw,pres_spec]}
      I:{allow:[D,A,sdk,http,db,fs],deny:[pres]}
      P:{allow:[A,D],deny:[I_direct]}
    wire:{comp_root:true,svc_loc:false,god:false,max_resp:1,reexport_modules:false,canonical_owner_required:true,owner_registry_required:true}

  change:
    scope:MIN
    require_trigger:"DO_CHANGE"
    ref_only_if:[fail_test,rule_violation,explicit]
    rename_req:[refs,tests,docs]
    allow_paths:[src/**,tests/**,docs/**,scripts/**]
    restrict_paths:[freqtrade/**,.github/**,docker/**,compose/**,pyproject.toml,requirements*.txt]
    restrict_paths_override_requires_approval:true

  ai_comments:{req:true,scope:[src/**,tests/**,scripts/**],rule:["new_code=>intent","if_file_no_comments=>retrofit","comments=WHY>WHAT"]}
  security:{no_secrets_code:true,no_secrets_logs:true,src:[env,secret_files],bypass:false}
  sdk:{broker:ICICI_BREEZE,adapter_boundary:true,undoc:FORBIDDEN}
  test:{unit:{net:false,fs:false,det:true,glob:false},int:{sdk:mock,sqlite:temp},regr:REQ}
  net:{bind:"127.0.0.1",ports:{st:8501,h:8000,nv:6080,v:5900},open:false,approval:true,docs:[deploy,compose,security]}

# === TERMINAL: keep prefixes minimal; rely on unix_prefix matching ===

  terminal:
    auto_exec:
      unix_prefix:[
        "python","pytest",".venv/bin/pytest","ruff","mypy","pydeps","pip check","pip list","pip freeze","pip show",
        "git status","git diff","git show","git log","git branch","git rev-parse",
        "rg","fd","ls","tree","wc","sed","awk","env","which","sha256sum","md5sum","jq","uname","df","free","ulimit","ps",
        "freqtrade --version","freqtrade list-exchanges","freqtrade list-markets","freqtrade download-data --dry-run","freqtrade trade --dry-run",
        "bash scripts/accept_all.sh","bash scripts/run_tests.sh","bash scripts/ci_verify.sh","bash scripts/collect_p12_data.sh","bash scripts/lint_check.sh","bash scripts/dry_run.sh"
      ]
    deny:
      unix_prefix:[
        "bash -c","sh","source",". ",
        "sudo","su",
        "rm","mv","cp","chmod","chown",
        "curl","wget","ssh","scp","rsync",
        "pip install","pip uninstall","pip upgrade","pip install -r","pipx","apt","apt-get","dpkg","snap","brew",
        "git commit","git push","git pull","git fetch","git reset","git rebase","git checkout","git clean",
        "freqtrade trade","freqtrade webserver","freqtrade rpc","freqtrade telegram","freqtrade install-ui","freqtrade download-data","freqtrade backtesting","freqtrade hyperopt","freqtrade new-strategy","freqtrade create-userdir",
        "&&","||",";","|","`","$(","<",">","2>","1>","&>"
      ]

  ownership:
    def:BLOCK
    objective:ONE_OWNER_PER_GLOBAL_ENTITY
    scope:[class,singleton,global_config,constant,enum,registry,service_locator,stateful_cache]
    rules:{single_owner:{block:true},no_reexports:{block:true},no_multi_singleton:{block:true},import_contract:{block:true}}
    allowed:[TYPE_CHECKING,owner_factory,explicit_DI]
    evidence_on_block:[entity,owner_candidate,dups,reexports,instantiations]

  sequencing:
    def:BLOCK
    objective:DETERMINISTIC_LIFECYCLE_ORDER
    no_side_effects_on_import:true
    single_orchestrator_required:true
    orchestrator_path_hint:"main.py"
    invariants:["config_before_net","logging_before_bg","rl_before_clients","ex_before_loop","loop_after_ready","shutdown_reverse_cleanup"]
    forbidden:["net_on_import","threads_on_import","boot_singletons_on_import","hidden_top_level_init"]
    exception_requires_approval: sequencing_exception

  time_determinism:
    def:BLOCK
    objective:DETERMINISTIC_TIME
    scope_paths:[src/**,tests/**,adapters/**]
    runtime_time_deps_must_inject:[now_fn,sleep_fn]
    forbidden_in_tests:["time.sleep(","asyncio.sleep(","datetime.now(","time.time(","perf_counter("]
    required_patterns:["FakeClock","now_fn","sleep_fn"]
    exception_requires_approval: time_determinism_exception

  exit_policy:
    def:BLOCK
    objective:EXITS_NO_SHORTS
    rules:{buyer_only:true,exit_intent_keys:[reduceOnly,ft_exit],sell_requires_exit_intent_if_no_pos:true,forbid_shorts:true,forbid_silent_bypass:true}
    exception_requires_approval: exit_policy_exception

  import_side_effect_scan:
    def:BLOCK
    objective:NO_IMPORT_SIDE_EFFECTS
    scope_paths:[src/**,adapters/**,freqtrade/**]
    forbid_patterns:["BreezeConnect\\(","requests\\.","httpx\\.","aiohttp\\.","websocket\\.","Thread\\(","create_task\\(","schedule\\.","patch_ccxt\\("]
    allow_if:["inside_def_or_class_only","guarded_by_if_name_main"]
    exception_requires_approval: import_side_effect_exception
```

## AG_WORKSPACE

```yaml
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
```
