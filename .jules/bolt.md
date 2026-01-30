## 2026-01-30 - Hidden Complexity in Properties
**Learning:** Properties like `self.timeframes` in `freqtrade`'s Exchange class may regenerate lists/dicts on every access. Using them in tight loops causes significant overhead ($O(N)$ lookup + allocation) compared to cached variables.
**Action:** Always inspect property implementations before using them in loops. Hoist invariant property accesses out of loops and convert to sets for membership tests.
