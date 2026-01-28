#!/usr/bin/env python3
"""
P20 Scan Port Exposure
Scans repository for risky network binding patterns:
1. "0.0.0.0" strings (bind all interfaces).
2. Docker port mappings without 127.0.0.1 prefix (e.g. "8080:8080").
"""

import os
import re
import sys
from pathlib import Path

# Paths to ignore (e.g. tests, lockfiles)
IGNORES = [
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "poetry.lock",
    "package-lock.json",
    "user_data/logs",
]

# Known exceptions (Allowlist)
# Format: "file_path": ["reason/comment"]
ALLOWLIST = {
    "docker/docker-compose-jupyter.yml": [
        "ip: 0.0.0.0",  # Container internal bind, mapped to 127.0.0.1 on host
    ],
    "docs/P20_API_SAFETY.md": [
        "0.0.0.0"  # Documentation mention of forbidden pattern
    ],
    "docs/PHASE_P20.md": [
        "0.0.0.0"  # Scope definition
    ],
    "scripts/ops/p20_scan_port_exposure.py": [
        "0.0.0.0"  # Self-reference
    ],
    # The prompt mentioned 6080 is external/allowed
    "docs/OPS_RUNBOOK.md": ["6080"],
    "scripts/gates/p20_no_open_ports_pos.sh": ["0.0.0.0"],
    "docs/utils.md": ["0.0.0.0"],
    "docs/rest-api.md": ["0.0.0.0", "8080:8080"],
    "tests/rpc/test_rpc_apiserver.py": ["0.0.0.0"],
    "freqtrade/configuration/deploy_config.py": ["0.0.0.0"],
    "scripts/ops/p20_scan_port_exposure.py": ["0.0.0.0", "8080:8080"],
}

RISKY_PATTERNS = [
    (r"0\.0\.0\.0", "Possible Bind to All Interfaces"),
    (r'("|\s)\d{4,5}:\d{4,5}', "Potential Public Port Mapping (missing 127.0.0.1)"),
]


def is_ignored(path):
    for ignore in IGNORES:
        if ignore in str(path):
            return True
    return False


def scan_file(file_path):
    violations = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                # Check for risky patterns
                for pattern, desc in RISKY_PATTERNS:
                    if re.search(pattern, line):
                        # Check allowlist
                        allowed = False
                        rel_path = str(file_path.relative_to(os.getcwd()))
                        if rel_path in ALLOWLIST:
                            for reason in ALLOWLIST[rel_path]:
                                if (
                                    reason in line
                                    or pattern in reason
                                    or (pattern == r"0\.0\.0\.0" and "0.0.0.0" in reason)
                                ):
                                    # Simple logical check for now
                                    allowed = True

                        if not allowed:
                            violations.append(f"{file_path}:{i} - {desc}: {line.strip()}")
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
    return violations


def main():
    root_dir = Path(os.getcwd())
    all_violations = []

    print(f"Scanning {root_dir} for port exposure risks...")

    for root, dirs, files in os.walk(root_dir):
        # Prune ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORES]

        for file in files:
            file_path = Path(root) / file
            if is_ignored(file_path):
                continue

            # Scan specific file types or all text files?
            # Let's verify commonly changed files: .py, .json, .yml, .sh, .md
            if file_path.suffix in [".py", ".json", ".yml", ".yaml", ".sh", ".md", ".dockerfile"]:
                all_violations.extend(scan_file(file_path))

    if all_violations:
        print("\n[FAIL] Risky Port Exposure Patterns Found:")
        for v in all_violations:
            print(v)
        sys.exit(1)
    else:
        print("\n[OK] No unchecked port exposure patterns found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
