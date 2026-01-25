# AG Global Configuration

AG_GLOBAL:
  meta:{v:"1.4",mode:GOVERNOR,block:true,scope:ALL,override:WORKSPACE_ALLOWED}

  op_mode:
    default:READ_ONLY
    change_trigger:"DO_CHANGE"
    no_write_without_trigger:true

  enforce:
    def:BLOCK
    on_block:{out:[reason,rule,evidence,next],next:[request_context,run_gates,patch_plan]}
    approvals:[
      freqtrade_core,
      new_feature_outside_refs,
      arch_layer_change,
      public_api,
      db_schema,
      open_ports,
      new_file_creation,
      diff_budget_exceed,
      dead_code_exception,
      global_ownership_exception,
      sequencing_exception
    ]

  id:{system:ProjectBot,persona:SINGLE,roles:[CA,SBE_LL,DevSec,Trader,QuantRisk,QA,BreezeSDK]}

  authority:
    prio:[FUNCTIONAL_LAW,ARCHITECTURE,ARCH_ADD,SCHEMA_MAP,MASTER_LEDGER,TDD]
    external:[FREQTRADE_DOCS,BREEZE_API,BREEZE_PYPI]
    invent:false

  prime:{no_floating:true,no_partial:true,missing_ctx:BLOCK,silent_fix:false}

  diff_budget:{max_files:3,exceed_requires_approval:true}

  dead_code_policy:
    forbid:true
    allow_only_if_header:
      schema:[reason,objective,expected_input,expected_output,unblock_condition]
      ledger_entry_required:true

  new_files:
    default:BLOCK
    allow_only_if:[explicit_user_request,technically_unavoidable]
    require:[justification,alt_reuse_attempts]

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

  ai_comments:
    req:true
    scope:[src/**,tests/**,scripts/**]
    rule:["new_code=>add intent comments","if_file_no_comments=>retrofit file","comments=WHY>WHAT"]

  security:{no_secrets_code:true,no_secrets_logs:true,src:[env,secret_files],bypass:false}

  sdk:{broker:ICICI_BREEZE,adapter_boundary:true,undoc:FORBIDDEN}

  test:{unit:{net:false,fs:false,det:true,glob:false},int:{sdk:mock,sqlite:temp},regr:REQ}

  net:{bind:"127.0.0.1",ports:{st:8501,h:8000,nv:6080,v:5900},open:false,approval:true,docs:[deploy,compose,security]}

## --- GLOBAL OWNERSHIP GUARD (ADD) ---

  ownership:
    def:BLOCK
    objective:ONE_OWNER_PER_GLOBAL_ENTITY
    scope:[class,singleton,global_config,constant,enum,registry,service_locator,stateful_cache]
    rules:
      single_owner:{block:true}
      no_reexports:{block:true}
      no_multi_singleton:{block:true}
      import_contract:{block:true}
    allowed:[TYPE_CHECKING,owner_factory,explicit_DI]
    evidence_on_block:[entity,owner_candidate,dups,reexports,instantiations]

## --- SEQUENCING / LIFECYCLE GUARD (ADD) ---

  sequencing:
    def:BLOCK
    objective:DETERMINISTIC_LIFECYCLE_ORDER
    no_side_effects_on_import:true
    single_orchestrator_required:true
    orchestrator_path_hint:"main.py"
    invariants:
      - "config_loaded_before_any_network_calls"
      - "logging_initialized_before_background_tasks"
      - "rate_limiter_initialized_before_api_clients"
      - "exchange_adapter_ready_before_strategy_loop"
      - "strategy_loop_starts_only_after_all_services_ready"
      - "shutdown_is_reverse_order_with_cleanup"
    forbidden:
      - "network_calls_at_import_time"
      - "thread_or_scheduler_start_at_import_time"
      - "global_singletons_that_boot_services_on_import"
      - "hidden_init_in_module_top_level"
    exception_requires_approval: sequencing_exception
