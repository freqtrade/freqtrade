#!/usr/bin/env python3
"""
Config Diff — Compare two GA YAML configs and show only differences.

Usage:
    python genetic_algorithm/scripts/config_diff.py config_a.yaml config_b.yaml
    python genetic_algorithm/scripts/config_diff.py config_a.yaml config_b.yaml --compact
    python genetic_algorithm/scripts/config_diff.py config_a.yaml config_b.yaml --json

Shows parameter-level differences in a clean table format.
Useful for reviewing what changed between wave experiments.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict to dot-separated keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key))
        elif isinstance(v, list):
            # Keep short lists as-is, flatten long ones
            if all(isinstance(x, dict) for x in v):
                for i, item in enumerate(v):
                    items.update(flatten_dict(item, f"{key}[{i}]"))
            else:
                items[key] = v
        else:
            items[key] = v
    return items


def compare_configs(config_a: Dict, config_b: Dict) -> List[Tuple[str, Any, Any, str]]:
    """Compare two configs and return list of (key, val_a, val_b, change_type) tuples."""
    flat_a = flatten_dict(config_a)
    flat_b = flatten_dict(config_b)

    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))

    diffs = []
    for key in all_keys:
        val_a = flat_a.get(key, "<absent>")
        val_b = flat_b.get(key, "<absent>")

        if val_a == val_b:
            continue

        if val_a == "<absent>":
            change_type = "ADDED"
        elif val_b == "<absent>":
            change_type = "REMOVED"
        else:
            change_type = "CHANGED"

        diffs.append((key, val_a, val_b, change_type))

    return diffs


def format_value(v: Any) -> str:
    """Format a value for display."""
    if isinstance(v, list):
        return str(v)
    if isinstance(v, bool):
        return str(v).lower()
    if v is None:
        return "null"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Compare two GA YAML configs")
    parser.add_argument("config_a", type=str, help="First config file (reference)")
    parser.add_argument("config_b", type=str, help="Second config file (comparison)")
    parser.add_argument("--compact", action="store_true", help="Compact output (skip storage/logging/output)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--section", type=str, help="Only show diffs for a specific section (e.g., genetic_algorithm)")
    args = parser.parse_args()

    path_a = Path(args.config_a)
    path_b = Path(args.config_b)

    if not path_a.exists():
        print(f"ERROR: Config not found: {path_a}")
        sys.exit(1)
    if not path_b.exists():
        print(f"ERROR: Config not found: {path_b}")
        sys.exit(1)

    with open(path_a) as f:
        config_a = yaml.safe_load(f)
    with open(path_b) as f:
        config_b = yaml.safe_load(f)

    diffs = compare_configs(config_a, config_b)

    # Filter by section if requested
    if args.section:
        diffs = [d for d in diffs if d[0].startswith(args.section)]

    # Compact mode: skip non-interesting sections
    if args.compact:
        skip_prefixes = ("storage.", "logging.", "output.", "hall_of_fame.directory",
                         "hall_of_fame.enabled", "overfit_analysis.")
        diffs = [d for d in diffs if not any(d[0].startswith(p) for p in skip_prefixes)]

    if not diffs:
        print("No differences found.")
        return

    if args.json:
        result = []
        for key, val_a, val_b, change_type in diffs:
            result.append({
                "parameter": key,
                "config_a": val_a,
                "config_b": val_b,
                "change": change_type,
            })
        print(json.dumps(result, indent=2, default=str))
        return

    # Text table output
    name_a = path_a.stem
    name_b = path_b.stem

    print()
    print(f"  Config Diff: {name_a} vs {name_b}")
    print(f"  {'=' * 70}")
    print()

    # Group by section
    sections: Dict[str, List] = {}
    for key, val_a, val_b, change_type in diffs:
        section = key.split(".")[0] if "." in key else "(root)"
        if section not in sections:
            sections[section] = []
        sections[section].append((key, val_a, val_b, change_type))

    for section, section_diffs in sections.items():
        print(f"  ── {section} ──")

        key_width = max(len(d[0]) for d in section_diffs)
        key_width = min(key_width, 45)

        for key, val_a, val_b, change_type in section_diffs:
            display_key = key[len(section)+1:] if key.startswith(section + ".") else key
            va = format_value(val_a)
            vb = format_value(val_b)

            if change_type == "ADDED":
                symbol = "+"
            elif change_type == "REMOVED":
                symbol = "-"
            else:
                symbol = "~"

            print(f"    {symbol} {display_key:<{key_width}}  {va:>20} → {vb:<20}")

        print()

    # Summary
    added = sum(1 for d in diffs if d[3] == "ADDED")
    removed = sum(1 for d in diffs if d[3] == "REMOVED")
    changed = sum(1 for d in diffs if d[3] == "CHANGED")
    print(f"  Summary: {changed} changed, {added} added, {removed} removed ({len(diffs)} total)")
    print()


if __name__ == "__main__":
    main()
