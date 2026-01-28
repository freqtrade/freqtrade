#!/usr/bin/env python3
"""
P19 Static Scanner for Exception Logging Compliance.
Scans python files in specified directories for 'except Exception' blocks
and ensures they log with stacktrace (logger.exception or exc_info=True).
"""

import ast
import sys
from pathlib import Path

VIOLATIONS = []


def check_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code, filename=filepath)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # We enforce this for broad exceptions typically, but the rule says "Exceptions".
            # Let's check if it catches 'Exception' or broad except.
            is_target_except = False
            if node.type is None:  # bare except
                is_target_except = True
            elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
                is_target_except = True

            if is_target_except:
                check_handler_compliance(node, filepath)


def check_handler_compliance(handler_node, filepath):
    # Check body for logger calls
    has_compliant_log = False

    for stmt in handler_node.body:
        # Check for logger.exception(...)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute):
                method_name = call.func.attr
                # Check for logger.exception or logging.exception
                if method_name == "exception":
                    has_compliant_log = True
                    break
                # Check for logger.error(..., exc_info=True)
                if method_name == "error":
                    for keyword in call.keywords:
                        if (
                            keyword.arg == "exc_info"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                        ):
                            has_compliant_log = True
                            break

    if not has_compliant_log:
        VIOLATIONS.append(
            f"{filepath}:{handler_node.lineno} - Except block missing logger.exception or exc_info=True"
        )


def main():
    dirs_to_scan = [Path("scripts"), Path("user_data/strategies")]

    print("P19: Scanning for Exception Logging Violations...")

    for d in dirs_to_scan:
        if d.exists():
            for p in d.rglob("*.py"):
                if "ops/p19_scan_exc_logging.py" in str(p):
                    continue
                check_file(p)

    if VIOLATIONS:
        print(f"FAILED: Found {len(VIOLATIONS)} violations:")
        for v in VIOLATIONS:
            print(v)
        sys.exit(1)
    else:
        print("SUCCESS: No violations found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
