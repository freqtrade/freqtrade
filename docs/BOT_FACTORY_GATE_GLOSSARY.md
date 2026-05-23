# Bot Factory Gate Glossary

Every gate in this glossary is a local artifact-review result. No gate name
starts paper, dry-run, live, canary, or exchange-facing execution.

| gate | permits | does not permit | next required gate |
| --- | --- | --- | --- |
| `initial_backtest_gate.pass` | candidate may proceed to historical walk-forward review | paper trading, dry-run trading, live trading, or exchange order placement | `walk_forward_gate.pass` |
| `eligible_for_walk_forward_review` | candidate may be evaluated across predefined historical windows | paper trading, dry-run trading, live trading, or exchange order placement | `walk_forward_gate.pass` |
| `REGIME_SCOPED_SELECTOR_ELIGIBLE` | local selector simulation may consider this candidate only inside eligible regimes | paper trading, dry-run trading, live trading, or process control | `paper_readiness.pass` |
| `GLOBAL_SELECTOR_ELIGIBLE` | local selector simulation may consider this candidate across declared regimes | paper trading, dry-run trading, live trading, or process control | `paper_readiness.pass` |
| `paper_readiness.pass` | a human may separately request a later no-startup paper plan | starting `freqtrade trade`, dry-run, paper, live, or canary processes | explicit human startup request plus Phase 3 startup preflight |
