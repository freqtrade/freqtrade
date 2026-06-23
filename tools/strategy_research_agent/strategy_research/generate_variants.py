#!/usr/bin/env python3
"""Generate isolated strategy variants from current research candidates."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
CANDIDATE_DIR = AGENT_ROOT / "candidates"
GENERATED_DIR = REPO_ROOT / "user_data/strategies/research_generated"
GENERATED_FILE = GENERATED_DIR / "generated_leverage_variants.py"
GENERATED_REGISTRY = AGENT_ROOT / "experiments/generated_variant_registry.json"
GENERATED_EXPERIMENT = AGENT_ROOT / "experiments/generated_leverage_variants_experiment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leverage",
        type=float,
        action="append",
        default=[3.0, 5.0, 10.0],
        help="Leverage cap for generated subclasses. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_candidates() -> list[dict[str, Any]]:
    candidates = []
    for path in sorted(CANDIDATE_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        candidates.append(item)
    return candidates


def class_suffix(value: float) -> str:
    text = str(value).replace(".", "p")
    return f"L{text}x"


def import_name(strategy_name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", strategy_name):
        raise ValueError(f"Unsafe strategy class name: {strategy_name}")
    return strategy_name


def generated_class_name(base: str, leverage: float) -> str:
    return f"Generated{base}{class_suffix(leverage)}"


def build_source(candidates: list[dict[str, Any]], leverages: list[float]) -> tuple[str, list[dict[str, Any]]]:
    bases = sorted({import_name(item["strategy"]) for item in candidates})
    imports = ", ".join(bases)
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        '"""Auto-generated isolated strategy variants.',
        "",
        "Do not edit by hand. Re-generate with user_data/strategy_research/generate_variants.py.",
        "These classes are research-only and must not be promoted to live without manual approval.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "from datetime import datetime",
        "from pathlib import Path",
        "",
        "sys.path.append(str(Path(__file__).resolve().parents[1]))",
        "",
        f"from btc_eth_risk_controlled_strategies import {imports}",
        "",
        "",
        f"GENERATED_AT_UTC = {generated_at!r}",
        "",
    ]
    registry_entries: list[dict[str, Any]] = []
    for item in candidates:
        base = import_name(item["strategy"])
        for leverage in leverages:
            cls = generated_class_name(base, leverage)
            lines.extend(
                [
                    "",
                    f"class {cls}({base}):",
                    f"    \"\"\"Research-only {leverage:g}x leverage-cap variant of {base}.\"\"\"",
                    "",
                    "    def leverage(",
                    "        self,",
                    "        pair: str,",
                    "        current_time: datetime,",
                    "        current_rate: float,",
                    "        proposed_leverage: float,",
                    "        max_leverage: float,",
                    "        entry_tag: str | None,",
                    "        side: str,",
                    "        **kwargs,",
                    "    ) -> float:",
                    f"        return min({leverage:.8g}, max_leverage)",
                    "",
                ]
            )
            registry_entries.append(
                {
                    "name": cls,
                    "base_strategy": base,
                    "family": f"generated-{item.get('family', 'unknown')}",
                    "source": "generated_from_research_candidate",
                    "hypothesis": f"Test whether {base} remains robust with a {leverage:g}x leverage cap.",
                    "risk_notes": "Generated in isolated research directory. Requires full backtesting, recursive-analysis, lookahead-analysis, and manual review before dry-run.",
                    "leverage_cap": leverage,
                }
            )
    return "\n".join(lines) + "\n", registry_entries


def main() -> None:
    args = parse_args()
    candidates = load_candidates()
    if not candidates:
        raise SystemExit("No candidate strategy files found.")
    source, registry_entries = build_source(candidates, args.leverage)
    if args.dry_run:
        print(source)
        print(json.dumps(registry_entries, indent=2, ensure_ascii=False))
        return

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FILE.write_text(source, encoding="utf-8")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_candidate_dir": str(CANDIDATE_DIR.relative_to(REPO_ROOT)),
        "generated_strategy_file": str(GENERATED_FILE.relative_to(REPO_ROOT)),
        "strategies": registry_entries,
    }
    GENERATED_REGISTRY.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment = {
        "id": "generated_leverage_variants",
        "title": "Generated leverage variants from current research candidates",
        "profile_ref": "strategy_registry.json",
        "strategy_path": "user_data/strategies/research_generated",
        "timeframes": [
            "1m"
        ],
        "timeranges": [
            "20240101-20260622"
        ],
        "fee": 0.0005,
        "strategies": [item["name"] for item in registry_entries],
        "checks": {
            "backtesting": True,
            "recursive_analysis": False,
            "lookahead_analysis": False
        },
        "notes": [
            "Generated variants are isolated research strategies.",
            "They require full checks and manual approval before dry-run."
        ],
    }
    GENERATED_EXPERIMENT.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {GENERATED_FILE.relative_to(REPO_ROOT)}")
    print(f"Wrote {GENERATED_REGISTRY.relative_to(REPO_ROOT)}")
    print(f"Wrote {GENERATED_EXPERIMENT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
