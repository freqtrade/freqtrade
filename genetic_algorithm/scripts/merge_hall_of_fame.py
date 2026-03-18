#!/usr/bin/env python3
"""
Merge Hall of Fame — Combine HoF archives from multiple wave experiments.

Creates a unified "exploration Hall of Fame" containing the best strategies
ever found across all experiments. Can be injected into future waves.

Usage:
    python genetic_algorithm/scripts/merge_hall_of_fame.py wave1
    python genetic_algorithm/scripts/merge_hall_of_fame.py wave1 --top 20
    python genetic_algorithm/scripts/merge_hall_of_fame.py wave1 wave2  # merge across waves
    python genetic_algorithm/scripts/merge_hall_of_fame.py wave1 --output merged_hof.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_hof(hof_dir: Path) -> List[Dict[str, Any]]:
    """Load Hall of Fame entries from a directory."""
    entries = []

    if not hof_dir.exists():
        return entries

    for f in sorted(hof_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                for entry in data:
                    entry['_source_file'] = str(f)
                    entries.extend(data if isinstance(data, list) else [data])
                    break
            elif isinstance(data, dict):
                data['_source_file'] = str(f)
                entries.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse {f}: {e}", file=sys.stderr)

    return entries


def get_fitness(entry: Dict[str, Any]) -> float:
    """Extract fitness score from a HoF entry."""
    if 'fitness' in entry:
        return float(entry['fitness'])
    if 'raw_fitness' in entry:
        return float(entry['raw_fitness'])
    if 'metrics' in entry and 'fitness' in entry['metrics']:
        return float(entry['metrics']['fitness'])
    return 0.0


def deduplicate(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate strategies based on gene hash or fitness+metrics combo."""
    seen = set()
    unique = []

    for entry in entries:
        # Create identity key from available fields
        key_parts = []

        if 'gene_hash' in entry:
            key_parts.append(entry['gene_hash'])
        elif 'strategy_id' in entry:
            key_parts.append(entry['strategy_id'])
        else:
            # Fall back to fitness + metrics fingerprint
            fitness = get_fitness(entry)
            metrics = entry.get('metrics', {})
            key_parts.append(f"{fitness:.6f}")
            key_parts.append(f"{metrics.get('profit', 0):.4f}")
            key_parts.append(f"{metrics.get('num_trades', 0)}")

        key = "|".join(str(k) for k in key_parts)

        if key not in seen:
            seen.add(key)
            unique.append(entry)

    return unique


def main():
    parser = argparse.ArgumentParser(description="Merge Hall of Fame across wave experiments")
    parser.add_argument("waves", nargs="+", help="Wave names (e.g., wave1 wave2)")
    parser.add_argument("--top", type=int, default=30, help="Keep top N entries (default: 30)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--min-fitness", type=float, default=0.0, help="Minimum fitness threshold")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_dir / "genetic_algorithm" / "data"

    all_entries = []
    sources = {}

    for wave in args.waves:
        # Find all HoF directories for this wave
        hof_dirs = sorted(data_dir.glob(f"hall_of_fame_{wave}_*"))

        if not hof_dirs:
            # Try alternate naming
            hof_dirs = sorted(data_dir.glob(f"hall_of_fame*{wave}*"))

        if not hof_dirs:
            print(f"  Warning: No HoF directories found for '{wave}'", file=sys.stderr)
            continue

        for hof_dir in hof_dirs:
            entries = load_hof(hof_dir)
            exp_name = hof_dir.name.replace("hall_of_fame_", "")

            for entry in entries:
                entry['_wave'] = wave
                entry['_experiment'] = exp_name

            all_entries.extend(entries)
            sources[hof_dir.name] = len(entries)

    if not all_entries:
        print("No Hall of Fame entries found.")
        sys.exit(0)

    # Deduplicate
    before_dedup = len(all_entries)
    all_entries = deduplicate(all_entries)
    after_dedup = len(all_entries)

    # Filter by minimum fitness
    if args.min_fitness > 0:
        all_entries = [e for e in all_entries if get_fitness(e) >= args.min_fitness]

    # Sort by fitness (descending)
    all_entries.sort(key=get_fitness, reverse=True)

    # Keep top N
    top_entries = all_entries[:args.top]

    # Output
    output_file = args.output or str(
        data_dir / f"hall_of_fame_exploration_{'_'.join(args.waves)}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(top_entries, f, indent=2, default=str)

    # Report
    print()
    print("  Hall of Fame Merge — Report")
    print("  " + "=" * 50)
    print()
    print("  Sources:")
    for src, count in sources.items():
        print(f"    {src}: {count} entries")
    print()
    print(f"  Total entries:     {before_dedup}")
    print(f"  After dedup:       {after_dedup}")
    print(f"  After filtering:   {len(all_entries)}")
    print(f"  Kept (top {args.top}):    {len(top_entries)}")
    print()

    if top_entries:
        print("  Top 5:")
        for i, entry in enumerate(top_entries[:5]):
            fitness = get_fitness(entry)
            wave = entry.get('_wave', '?')
            exp = entry.get('_experiment', '?')
            metrics = entry.get('metrics', {})
            profit = metrics.get('profit', 'N/A')
            print(f"    {i+1}. fitness={fitness:.4f}  profit={profit}  [{wave}/{exp}]")

    print()
    print(f"  Saved to: {output_file}")
    print(f"  To inject into future runs, copy to the target HoF directory")
    print()


if __name__ == "__main__":
    main()
