# AG Global Configuration

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
