from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SECRET_RE = re.compile(
    r"""(?ix)
    (api[_-]?key|secret|password|token|jwt_secret_key|ws_token)
    \s*[:=]\s*
    ["'][^"']{8,}["']
    """
)

DANGEROUS_FUNCTIONS = {
    "create_order",
    "create_limit_buy_order",
    "create_limit_sell_order",
    "create_market_buy_order",
    "create_market_sell_order",
    "private_post_order",
    "fapiPrivatePostOrder",
    "request_order",
}

INDICATOR_CONTEXT_PREFIXES = (
    "populate_",
    "feature_engineering_",
    "set_freqai_targets",
)


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    severity: str
    message: str


@dataclass
class SafetyReport:
    ok: bool
    files_checked: int
    findings: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "files_checked": self.files_checked,
            "findings": [asdict(f) for f in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []
        self._function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        func_name = self._call_name(node.func)
        attr_name = self._attr_name(node.func)

        if attr_name == "shift" and self._has_exact_negative_one_arg(node):
            self._add(node, "no_shift_minus_one", "error", "Exact shift(-1) is forbidden.")

        if func_name in {"requests.post", "httpx.post"}:
            self._add(
                node,
                "no_direct_order_api",
                "error",
                "Direct HTTP POST calls are not allowed in strategy code.",
            )

        if attr_name in DANGEROUS_FUNCTIONS or func_name in DANGEROUS_FUNCTIONS:
            self._add(
                node,
                "no_direct_exchange_order_call",
                "error",
                f"Direct exchange order call '{func_name or attr_name}' is forbidden.",
            )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if self._is_iloc_minus_one(node):
            if self._in_indicator_context():
                self._add(
                    node,
                    "no_iloc_minus_one_in_signal_generation",
                    "error",
                    "iloc[-1] is forbidden in indicator/entry/exit generation.",
                )
            else:
                self._add(
                    node,
                    "iloc_minus_one_review",
                    "warning",
                    "iloc[-1] found outside signal generation; review for lookahead risk.",
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id.lower()
        if name in {"future", "lookahead"}:
            self._add(
                node,
                "future_named_reference_review",
                "warning",
                f"Suspicious name '{node.id}'; review for lookahead risk.",
            )
        self.generic_visit(node)

    def _in_indicator_context(self) -> bool:
        if not self._function_stack:
            return False
        fn = self._function_stack[-1]
        return any(fn.startswith(prefix) for prefix in INDICATOR_CONTEXT_PREFIXES)

    def _has_exact_negative_one_arg(self, node: ast.Call) -> bool:
        if node.args and self._is_negative_one(node.args[0]):
            return True
        return any(
            keyword.arg == "periods" and self._is_negative_one(keyword.value)
            for keyword in node.keywords
        )

    def _is_iloc_minus_one(self, node: ast.Subscript) -> bool:
        if not isinstance(node.value, ast.Attribute) or node.value.attr != "iloc":
            return False
        return self._slice_contains_negative_one_row(node.slice)

    def _slice_contains_negative_one_row(self, node: ast.AST) -> bool:
        if self._is_negative_one(node):
            return True
        if isinstance(node, ast.Tuple) and node.elts:
            return self._slice_contains_negative_one_row(node.elts[0])
        if isinstance(node, ast.Slice):
            return self._is_negative_one(node.lower)
        return False

    def _is_negative_one(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Constant):
            return node.value == -1
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and node.operand.value == 1
        )

    def _attr_name(self, node: ast.AST) -> str:
        return node.attr if isinstance(node, ast.Attribute) else ""

    def _call_name(self, node: ast.AST) -> str:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    def _add(self, node: ast.AST, rule: str, severity: str, message: str) -> None:
        self.findings.append(
            Finding(
                path=str(self.path),
                line=getattr(node, "lineno", 0),
                rule=rule,
                severity=severity,
                message=message,
            )
        )


def scan_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []

    for match in SECRET_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        findings.append(
            Finding(
                path=str(path),
                line=line,
                rule="no_hardcoded_secret",
                severity="error",
                message="Potential hardcoded secret or credential.",
            )
        )

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        findings.append(
            Finding(
                path=str(path),
                line=exc.lineno or 0,
                rule="syntax_error",
                severity="error",
                message=str(exc),
            )
        )
        return findings

    visitor = _SafetyVisitor(path)
    visitor.visit(tree)
    findings.extend(visitor.findings)
    return findings


def iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts))
    return files


def scan_paths(paths: list[Path]) -> SafetyReport:
    files = iter_python_files(paths)
    findings: list[Finding] = []
    for file_path in files:
        findings.extend(scan_file(file_path))

    ok = not any(f.severity == "error" for f in findings)
    return SafetyReport(ok=ok, files_checked=len(files), findings=findings)
