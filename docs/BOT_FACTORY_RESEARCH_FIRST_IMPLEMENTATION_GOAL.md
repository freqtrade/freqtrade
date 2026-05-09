# Bot Factory Research-First Implementation Goal

このファイルは、ローカルの Codex CLI に `/goal` で渡すための作業指示です。

この goal の目的は、Freqtrade 戦略候補をさらに量産することではありません。過去 32 件の失敗結果を前提に、**取引コスト控除後に残る市場効果だけを発見・反証し、条件を満たす thesis がない場合は strategy candidate を生成しない** 研究・評価ゲートを実装してください。

---

## 0. 最重要原則

- `paper_ready_count=0` は、この goal では失敗とは限らない。
- post-cost edge のある thesis が見つからない場合は、`no candidate generated` を明記して成功扱いにする。
- 新しい strategy candidate の生成は目的ではない。
- strategy codegen は、Research Lab / Edge Discovery の gate をすべて通過した場合にだけ許可する。
- parameter-only retry、threshold loosening、indicator variant 量産、FreqAI black-box retry は禁止。
- 過去に失敗した thesis ID / mechanism class の再利用は禁止。
- 実運用注文、live trading、secret の変更、API key の出力は絶対に行わない。

---

## 1. 前提となる失敗結果

前回の Bot Factory /goal では、研究・評価・失敗記憶インフラは増えたが、利益候補、walk-forward 通過候補、paper-ready 候補は作れなかった。

重要な集計:

- `candidate_count=32`
- `paper_ready_count=0`
- `zero_trade_count=7`
- `negative_return_count=22`
- `walk_forward_failed_count=32`
- `parameter_only_retry_allowed=false`

代表的な失敗:

- Mark Discount Reclaim:
  - `total_return_pct=-37.832122396`
  - `profit_factor=0.244072592568329`
  - `entry_signal_count=9603`
  - sparse signal ではなく、entry 後の edge が悪い。
- Mark Fair-Value Momentum Lag:
  - 初期 local screen: `net_edge_bps=15.99047`
  - closed-context alignment 修正後: `net_edge_bps=-11.04985`
  - `profitable_windows_ratio=0.0`
  - local screen と Freqtrade semantics のズレによる false positive。
- Directional Change Overshoot:
  - `total_return_pct=-3.2605379719999994`
  - `profit_factor=0.3464660896687346`
  - `walk-forward pass_rate=0.0`
- BTC/ETH relative value / cross-asset:
  - `net_edge_bps=-11.699275`
  - `profitable_windows_ratio=0.25`
- Volume-clock liquidity momentum:
  - `expected_edge_bps=0.255677`
  - `all_in_cost_bps=12.0`
  - `net_edge_bps=-11.744323`
- funding-neutral impulse drift:
  - `net_edge_bps=-14.205144`

観察された中心問題:

- `all_in_cost_bps=12.0` を超える edge がほぼ出ていない。
- positive に見えた local screen が、closed-context alignment や generated strategy の entry semantics に合わせると消える。
- walk-forward robustness が極端に弱い。
- strategy variant を増やしても negative evidence が増えただけで、positive evidence は増えなかった。

---

## 2. ブランチ選定は実装内容を確認してから決める

作業前に、以下 2 つの候補ブランチを必ず比較する。

- 以前の作業ブランチ: `codex/bot-factory-candidate-factory-completion`
- 統合ブランチ: `develop`

最初からどちらかに決め打ちしない。必ず差分と実装状況を確認してから、作業 base と PR target を決定する。

### 2.1 必須確認コマンド

破壊的なコマンドは禁止。`git reset --hard`、`git clean -fd` は使わない。

```bash
git fetch --all --prune
git status --short
git branch --show-current
git branch --list "develop" "codex/bot-factory-candidate-factory-completion"
git log --oneline --decorate --graph --max-count=30 --all
git diff --stat develop..codex/bot-factory-candidate-factory-completion
git diff --name-status develop..codex/bot-factory-candidate-factory-completion
```

存在すれば以下も確認する。

```bash
git show develop:docs/BOT_FACTORY_GOAL_AUDIT.md >/tmp/develop_goal_audit.md 2>/dev/null || true
git show codex/bot-factory-candidate-factory-completion:docs/BOT_FACTORY_GOAL_AUDIT.md >/tmp/codex_goal_audit.md 2>/dev/null || true

rg -n "all_in_cost_bps|failure synthesis|causal failure|rejection memory|Edge Discovery|local falsification|mechanism class|parameter-only|closed-context|paper_ready_count" docs scripts tests freqtrade user_data 2>/dev/null || true
```

### 2.2 ブランチ選定ルール

#### A. 以前の作業ブランチを base にする条件

以下に該当する場合は、`codex/bot-factory-candidate-factory-completion` を base にして新ブランチを切る。

- `develop` に Bot Factory の failure memory / causal map / Edge Discovery / local falsification / rejection memory / parameter-only retry block が未導入。
- 今回の作業が、以前の branch で追加された Research Lab / Edge Discovery / gate 実装を直接拡張する必要がある。
- 以前の branch の実装が coherent で、関連テストが実行可能。
- `develop` から始めると大部分のインフラを再実装することになる。

この場合:

- 作業 branch: `codex/bot-factory-research-first-edge-gates`
- base: `codex/bot-factory-candidate-factory-completion`
- PR target: `codex/bot-factory-candidate-factory-completion`
- 理由: stacked PR として、既存の大きい差分に新規差分を混ぜないため。

#### B. `develop` を base にする条件

以下に該当する場合は、`develop` を base にして新ブランチを切る。

- `develop` がすでに以前の branch の主要インフラを含んでいる、または同等以上の実装に置き換わっている。
- 以前の branch が大きすぎる、壊れている、未整理、または今回の作業に不要な speculative code を大量に含む。
- 今回の作業を `develop` 上で小さく self-contained に実装できる。

この場合:

- 作業 branch: `codex/bot-factory-research-first-edge-gates`
- base: `develop`
- PR target: `develop`

#### C. 両方が部分的な場合

以下の場合は、`develop` を base にして、以前の branch から最小限の必要コンポーネントだけを port / cherry-pick / 再実装する。

- 以前の branch に有用な gate や docs はあるが、そのまま base にすると巨大 PR になる。
- `develop` に一部実装があり、以前の branch の一部だけが必要。
- 以前の branch の生成物や strategy candidates を持ち込む必要がない。

この場合:

- 作業 branch: `codex/bot-factory-research-first-edge-gates`
- base: `develop`
- PR target: `develop`
- 以前の branch から取り込んだものは `docs/BOT_FACTORY_BRANCH_DECISION.md` と PR body に明記する。

### 2.3 ブランチ判断ドキュメント

実装前に必ず以下を作成する。

`docs/BOT_FACTORY_BRANCH_DECISION.md`

必須項目:

```markdown
# Bot Factory Branch Decision

## Compared branches

- develop
- codex/bot-factory-candidate-factory-completion

## Summary of existing implementation on develop

## Summary of existing implementation on codex/bot-factory-candidate-factory-completion

## Relevant files compared

## Selected base branch

## Selected PR target branch

## Reason

## Risks

## What will not be included in this PR
```

---

## 3. Goal file の扱い

この goal file がまだ repository に存在しない場合は、選定した作業 branch 上で以下に保存する。

`docs/BOT_FACTORY_RESEARCH_FIRST_IMPLEMENTATION_GOAL.md`

このファイル自体も PR に含める。ただし、branch 選定前に不用意に commit しない。

---

## 4. 実装対象

既存の module / script / docs の構成を先に確認し、既存 style に合わせて実装する。ファイル名や module 名は既存構造に合わせてよいが、以下の機能は必須。

---

### 4.1 Cost Model Audit

目的:

固定値 `all_in_cost_bps=12.0` に依存せず、best / normal / stress の 3 段階で post-cost edge を評価できるようにする。

実装要件:

- cost scenario を表現する構造を追加する。
- 最低限、以下を表現できるようにする。
  - `scenario_name`
  - `fee_bps_entry`
  - `fee_bps_exit`
  - `spread_bps`
  - `slippage_bps_entry`
  - `slippage_bps_exit`
  - `adverse_selection_bps`
  - `no_fill_rate`
  - `partial_fill_rate`
  - `exit_taker_rate`
  - `stress_multiplier`
  - `total_cost_bps`
- pair / timeframe / order type / liquidity tier / volatility regime で cost を切り替えられるようにする。
- default scenario は以下の 3 種を持つ。
  - `best`
  - `normal`
  - `stress`
- `normal` は既存の `all_in_cost_bps=12.0` と整合するように初期化してよい。
- `stress` は fee/slippage を 1.5x 以上にした保守的評価を可能にする。
- maker 前提の評価では、必ず no-fill / partial-fill / adverse selection を扱う。
- cost model は docs と test を持つ。

成果物:

- `docs/BOT_FACTORY_COST_MODEL_AUDIT.md`
- cost model implementation
- cost model tests

---

### 4.2 Freqtrade Semantics-Aligned Event Study

目的:

Research Lab の local event study を、Freqtrade strategy と同じ entry / exit semantics で評価する。

実装要件:

- signal は candle close 後にのみ確定する。
- entry は原則として next candle open、または保守的な limit fill model とする。
- forward return に未来情報を混ぜない。
- mark price / funding / OI / order book / liquidation などの structural data は timestamp alignment を厳密に扱う。
- closed-context alignment を必須化する。
- overlapping events による過大評価を避ける。
- generated strategy の entry/exit semantics と local event study の semantics が一致していることを検証する。
- 古い positive artifact は再利用しない。

成果物:

- semantics-aligned event study implementation
- alignment validation tests
- `docs/BOT_FACTORY_EDGE_DISCOVERY_REPORT.md` に alignment 結果を出力する仕組み

---

### 4.3 Event-Level Post-Cost Edge Report

目的:

strategy codegen の前に、signal 単位で post-cost edge が存在するかを検証する。

必須 metric:

- `thesis_id`
- `mechanism_class`
- `event_count`
- `entry_signal_count`
- `gross_edge_bps`
- `cost_bps_best`
- `cost_bps_normal`
- `cost_bps_stress`
- `net_edge_bps_best`
- `net_edge_bps_normal`
- `net_edge_bps_stress`
- `profitable_windows_ratio`
- `walk_forward_pass_rate`
- `lower_confidence_bound_bps`
- `pair_concentration`
- `calendar_concentration`
- `holding_period`
- `negative_control_random_entry_delta_bps`
- `negative_control_shuffled_signal_delta_bps`
- `negative_control_shifted_signal_delta_bps`
- `passes_research_gate`
- `rejection_reason`

出力先:

- `docs/BOT_FACTORY_EDGE_DISCOVERY_REPORT.md`
- 既存の JSON / summary artifact があれば、同じ形式にも出力する。

---

### 4.4 Negative Controls

目的:

false positive、timestamp leakage、market beta、偶然を排除する。

最低限、以下 3 つを実装する。

1. Random entry control
   - 同じ trade count / holding period に近い random entry と比較する。
2. Shuffled signal control
   - signal の時系列を shuffle して比較する。
3. Shifted signal control
   - signal を過去・未来にずらして比較する。
   - shifted signal でも同程度に勝つ場合は alignment / leakage 疑いとして reject する。

合格条件:

- real signal が各 negative control に明確に勝つ。
- 差分 metric を report に出す。
- control に勝てない thesis は codegen 禁止。

---

### 4.5 Research Gate / Candidate Gate

目的:

post-cost edge が確認できない thesis を strategy candidate 化しない。

最低合格基準:

- `net_edge_bps_normal >= +6`
- 望ましくは `net_edge_bps_normal >= +12`
- `net_edge_bps_stress > 0`
- `profitable_windows_ratio >= 0.7`
- `walk_forward_pass_rate >= 0.6`
- `lower_confidence_bound_bps > 0`
- single pair 依存ではない。
- single calendar window 依存ではない。
- negative controls に明確に勝つ。
- local event study と generated strategy の entry / exit semantics が一致する。
- 過去 failure mechanism class と重複しない。

reject 条件:

- `net_edge_bps_normal <= 0`
- gross edge が normal cost 未満。
- stress cost で negative。
- `profitable_windows_ratio < 0.7`
- `walk_forward_pass_rate < 0.6`
- lower confidence bound が 0 以下。
- negative control に勝てない。
- local screen と generated strategy semantics がズレている。
- parameter-only retry。
- threshold loosening。
- 失敗済み mechanism class の再利用。
- positive result が single pair / single window に依存。
- zero-trade / sparse-signal を threshold relaxation で無理に救済している。

成果物:

- gate implementation
- gate tests
- rejected thesis は failure memory / rejection memory に理由つきで記録する。

---

### 4.6 Research Thesis Selection

目的:

過去 failure を踏まえ、候補化する前に thesis を厳選する。

要件:

- 最新の failure synthesis / causal failure map / rejection memory を読む。
- 過去に失敗した thesis ID / mechanism class を除外する。
- research thesis は最大 3 件まで。
- thesis ごとに causal hypothesis を記録する。
- 少なくとも以下のカテゴリを優先して検討する。
  - low-turnover / high-timeframe regime strategy
  - no-trade filter as alpha
  - liquidation / forced-flow rare events
  - funding / basis as carry, not directional alpha
  - execution alpha with conservative maker fill model
- ただし、合格 gate を満たさない場合は strategy candidate を生成しない。

成果物:

- `docs/BOT_FACTORY_NEXT_RESEARCH_PLAN.md`
- no passing thesis の場合は、`no candidate generated` を明記する。

---

### 4.7 Paper Promotion Checklist

目的:

paper-ready の条件を明文化し、false positive を paper に進ませない。

paper-ready 条件:

- post-cost edge positive。
- `net_edge_bps_normal >= +6`
- 望ましくは `net_edge_bps_normal >= +12`
- stress cost でも positive。
- `profitable_windows_ratio >= 0.7`
- `walk_forward_pass_rate >= 0.6`
- lower confidence bound > 0。
- negative controls に勝つ。
- lookahead / recursive / semantics alignment が clean。
- single pair / single month 依存ではない。
- risk overlay を入れても expectancy が残る。
- dry / paper で 30〜60 日または十分な trade 数を確認するまでは live に進めない。

成果物:

- paper promotion checklist を docs に追加または既存 docs に統合する。

---

## 5. 禁止事項

以下は実装しない。

- 33 個目の strategy smoke を無条件に作ること。
- parameter tuning のみでの retry。
- threshold loosening。
- indicator variant farm。
- FreqAI black-box retry。
- DCA / martingale で backtest を良く見せること。
- leverage で profit % だけを大きく見せること。
- local screen positive だけで codegen すること。
- 古い positive artifact を再利用すること。
- generated artifacts / backtest output / caches / private datasets を Git に混ぜること。
- live trading / 実注文 / API secret の変更。

---

## 6. テスト要件

既存テスト構成を確認し、関連テストを追加・更新する。

最低限の test coverage:

- cost scenario の total cost 計算。
- best / normal / stress の切り替え。
- semantics-aligned next-candle entry。
- closed-context alignment。
- shifted signal control が leakage を検出できること。
- random / shuffled controls に勝てない thesis が reject されること。
- gate 合格条件を満たす synthetic positive case。
- gate 不合格条件を満たす synthetic negative case。
- parameter-only retry が reject されること。
- no passing thesis の場合に `no candidate generated` が report されること。

実行するテスト例:

```bash
python -m pytest tests/test_bot_factory.py -q
python -m pytest tests -q
```

既存プロジェクトの test command が異なる場合は、既存 docs / pyproject / Makefile に合わせる。

---

## 7. 成果物一覧

必須成果物:

- `docs/BOT_FACTORY_BRANCH_DECISION.md`
- `docs/BOT_FACTORY_RESEARCH_FIRST_IMPLEMENTATION_GOAL.md`
- `docs/BOT_FACTORY_COST_MODEL_AUDIT.md`
- `docs/BOT_FACTORY_EDGE_DISCOVERY_REPORT.md`
- `docs/BOT_FACTORY_NEXT_RESEARCH_PLAN.md`
- cost model implementation
- semantics-aligned event study implementation
- negative controls implementation
- research gate / candidate gate implementation
- updated failure memory / rejection memory
- tests

no passing thesis の場合:

- strategy candidate を生成しない。
- `docs/BOT_FACTORY_EDGE_DISCOVERY_REPORT.md` と PR body に `no candidate generated` を明記する。
- これは goal failure ではなく、false positive を防いだ successful rejection と扱う。

---

## 8. PR 作成

実装・テスト後、PR を作成する。

### 8.1 Commit

差分を確認する。

```bash
git status --short
git diff --stat
git diff --name-status
```

不要な生成物、cache、backtest output、private dataset が含まれていないことを確認する。

commit message 例:

```text
Add research-first Bot Factory edge gates
```

### 8.2 PR body

PR body は以下を含める。

```markdown
## Summary

- Added research-first Bot Factory edge discovery gates.
- Added cost model audit with best / normal / stress scenarios.
- Added Freqtrade semantics-aligned event study checks.
- Added negative controls and post-cost edge reporting.
- Strategy candidate generation remains blocked unless research gates pass.

## Branch decision

- Compared branches:
  - develop
  - codex/bot-factory-candidate-factory-completion
- Selected base:
- Selected PR target:
- Reason:

## Key behavior

- `paper_ready_count=0` is allowed when no thesis passes post-cost edge gates.
- No strategy candidate is generated unless the research gate passes.
- `no candidate generated` is treated as a valid successful rejection.

## Artifacts

- docs/BOT_FACTORY_BRANCH_DECISION.md
- docs/BOT_FACTORY_RESEARCH_FIRST_IMPLEMENTATION_GOAL.md
- docs/BOT_FACTORY_COST_MODEL_AUDIT.md
- docs/BOT_FACTORY_EDGE_DISCOVERY_REPORT.md
- docs/BOT_FACTORY_NEXT_RESEARCH_PLAN.md

## Tests

Paste executed commands and results.

## Risks / Follow-ups

- Cost model still needs live/paper calibration.
- Maker fill assumptions require conservative no-fill / adverse-selection validation.
- Passing research gate does not imply live profitability.
```

### 8.3 PR 作成コマンド

`gh` が利用可能なら、選定した PR target に対して PR を作成する。

```bash
gh pr create \
  --base "<SELECTED_PR_TARGET>" \
  --head "$(git branch --show-current)" \
  --title "Add research-first Bot Factory edge gates" \
  --body-file /tmp/bot_factory_research_first_pr_body.md
```

`gh` が使えない場合は、PR 作成用の正確なコマンド、現在の branch、base branch、差分 summary を最終出力に残す。

---

## 9. 最終出力

完了時に以下を出力する。

```markdown
## Completed

## Selected base branch

## Selected PR target branch

## PR URL

## Files changed

## Tests run

## Candidate generation result

- generated candidate: yes/no
- if no: no candidate generated
- reason:

## Important findings

## Remaining risks
```

PR を作成できた場合は PR URL を出す。作成できなかった場合は、その理由と再実行可能な `gh pr create` コマンドを出す。
