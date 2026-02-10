## 2026-01-30 - Hidden Complexity in Properties
**Learning:** Properties like `self.timeframes` in `freqtrade`'s Exchange class may regenerate lists/dicts on every access. Using them in tight loops causes significant overhead ($O(N)$ lookup + allocation) compared to cached variables.
**Action:** Always inspect property implementations before using them in loops. Hoist invariant property accesses out of loops and convert to sets for membership tests.

## 2026-02-10 - Dataframe Construction & Sorting Overhead
**Learning:** Constructing DataFrames from a list of lists is significantly slower than from a dictionary of NumPy arrays due to type inference and validation. Also, `sort_values` is expensive ($O(N \log N)$); checking `is_monotonic_increasing` ($O(N)$) first avoids redundant sorting for time-series data.
**Action:** Use dict-of-arrays for DataFrame construction in hot paths. Always check sortedness before sorting if data is likely pre-sorted.
