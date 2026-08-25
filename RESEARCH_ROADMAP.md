# Research Program — Combined Release Calendar

> Single execution-order calendar spanning both research proposals in this repo:
> `TRADER_WALLET_MINING_PROPOSAL.md` (WM) and `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` (SD).
> Neither proposal's own release numbering runs across the other, and SD had no release
> numbering at all until this doc -- this is the merge point. Sequencing below is the
> user's own approved reordering (2026-08-25): don't finish either subsystem before
> connecting them: close the wallet-mining validity gap, make strat-discovery's engine
> pluggable, wire the two together on one manual example, and only then decide what else
> either subsystem actually needs.

## Numbering key

- **WM-R#** -- wallet-mining's own release numbers, unchanged from
  `TRADER_WALLET_MINING_PROPOSAL.md`'s "Phased release plan" section.
- **SD-R#** -- strat-discovery release numbers, assigned here for the first time (that
  proposal has never had a release breakdown, only its original §1-20 phases).
- **BRIDGE-R#** -- new work items that are neither proposal's own -- specifically, an
  earlier and smaller slice of what WM's own Release 7 eventually becomes at full scale
  (see the note under BRIDGE-R1 below).

## Calendar

| # | Release | Subsystem | Status | Scope |
|---|---------|-----------|--------|-------|
| 1 | WM-R1 Hyperliquid ingestion | Wallet mining | ✅ Shipped (#16) | Read-only single-wallet fill ingestion |
| 2 | WM-R2 Trade reconstruction | Wallet mining | ✅ Shipped (#17) | Zero-crossing position-transition trade grouping |
| 3 | WM-R3 Performance metrics + report | Wallet mining | ✅ Shipped (#18) | `compute_metrics`/`format_report`, `trader-report` CLI |
| 4 | WM-R4 Temporal validation | Wallet mining | 🔨 In progress | TRAIN/VALIDATION/TEST/FORWARD split -- spec + plan written (`docs/superpowers/{specs,plans}/2026-08-25-trader-mining-release-4*`), implementation not yet started |
| 5 | SD-R1 StrategyHypothesis interface | Strat discovery | ⏳ Not started | Smallest useful common interface (identity/parameters/required_data/`generate_signals()`); migrate the 3 existing strategies (`EmaTrendFollow`, `MacdMomentum`, `BandtasticMeanReversion`) behind it -- moving a strategy behind the interface must not change its validated gate results |
| 6 | BRIDGE-R1 Connect wallet-mining → strat-discovery | Both | ⏳ Not started | One real WM-R4 wallet observation, turned into a **human-defined** hypothesis, expressed as a `StrategyHypothesis` (SD-R1), run through SD's existing `research/gate.py` OOS machinery. No automated hypothesis generation yet. This is a smaller, pulled-forward slice of what WM's own **Release 7** ("Hypothesis generation + backtester integration," proposal §8-9) describes at full scale -- proving the pipe connects end-to-end on one manual example, before WM-R5/R6 exist to feed it automatically. WM-R7 itself, at full scale, still needs WM-R5+WM-R6 as real inputs and stays scheduled after them (see below). **Before picking the wallet for this release's manual example**, check it against the `tid=0` bug below -- the fix is scheduled after this release, and per FIELD-NOTES.md a second same-wallet dust-conversion fill is silently dropped (not just a cross-wallet crash), so a single wallet isn't automatically immune |
| 7 | WM-R5 Multi-wallet + selection-bias guardrails | Wallet mining | ⏳ Not started (deferred) | Only once a concrete research question needs more than one wallet. Needs its own design pass for the frozen-cohort/pre-registered-inclusion-protocol gap flagged in the proposal's review notes -- not a rubber-stamp of the original text |
| 8 | WM-R6 Behavioral analysis | Wallet mining | ⏳ Not started (deferred) | Needs 2+ real candidate wallets from WM-R5 to compare -- a single wallet has nothing to compare against |
| 9 | SD-R2+ Remaining strategy families / crypto-specific features | Strat discovery | ⏳ Not started (deferred) | Breakout, cross-sectional momentum, funding/basis/liquidation research, universe construction, Monte Carlo -- add only when a concrete hypothesis (from BRIDGE-R1 or later) actually demands one, not because it's on the original proposal's list |
| 10 | WM-R7 Full hypothesis generation + backtester integration | Both | ⏳ Not started (deferred) | The proposal's original full-scale vision (§8-9) -- automated hypothesis generation from WM-R6's behavioral findings across the WM-R5 candidate cohort. Supersedes BRIDGE-R1's manual version once real automated inputs exist |

## Explicitly not scheduled yet

- WM's automated wallet discovery (proposal §5) and every non-Hyperliquid provider --
  out of scope in the original proposal, unchanged here.
- The `tid=0` dust-conversion sentinel-collision bug (`FIELD-NOTES.md`) -- tracked
  separately, still slated for "before or alongside WM-R5" (multi-wallet is where the
  collision risk becomes real, not before).
- SD's non-expert guided CLI workflow (proposal §16), live strategy health dashboards
  beyond what `research/health.py` already does, elaborate experiment-ledger tooling
  beyond `research/pbo.py`'s existing per-run trial counting.

## Why this order (already cross-checked)

WM-R4 before everything else: independently confirmed via lmchatbot (Gemini, verified by
ChatGPT) -- without it, every wallet-mining metric is lookahead-contaminated and unfit to
feed a hypothesis into anything. SD-R1 and BRIDGE-R1 before WM-R5/R6/SD-R2: proving the
full pipeline works end-to-end on one manual hypothesis is worth more than expanding either
subsystem's breadth first -- matches this doc's own header note on the user's approved
reordering.

## Review notes (lmchatbot sign-off, 2026-08-25)

This calendar (full text, all 10 releases) was reviewed by Gemini and cross-verified by
ChatGPT before being committed. Verdict: **"Sound with minor adjustments"** on the draft
pass; the verify pass then walked back 3 of the draft's 4 concerns as unfounded (SD-R1's
regression criterion, a proposed WM-R4 export-schema requirement, and the WM-R5/WM-R6
sequencing, each re-examined against this doc's own already-stated scope and found not to
be real gaps -- not carried forward here). One concern survived the verify pass and is
reflected in BRIDGE-R1's row above: the `tid=0` dust-conversion sentinel bug (FIELD-NOTES.md)
is fixed *after* BRIDGE-R1 in this calendar, but can silently drop fills for a single wallet
too, not just collide across wallets -- so BRIDGE-R1's chosen wallet needs an explicit check
against it before use, rather than assuming single-wallet scope makes the bug irrelevant.
