from __future__ import annotations

import importlib
import ast
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from importlib import metadata
from typing import Any, Callable, Sequence


FREQAI_LABEL_NOTICE = "FreqAI labels are backtest labels, not live trading instructions."


@dataclass(frozen=True)
class DependencySpec:
    import_name: str
    package_name: str | None = None
    required: bool = True


@dataclass
class DependencyStatus:
    name: str
    import_name: str
    package_name: str
    required: bool
    installed: bool
    version: str | None
    error: str | None


@dataclass
class FreqAIEnvironmentReport:
    ok: bool
    dependencies: list[DependencyStatus]
    python_version: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dependencies": [asdict(dependency) for dependency in self.dependencies],
            "python_version": self.python_version,
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class FreqAIColumnReference:
    path: str
    line: int
    function: str
    column: str


@dataclass
class FreqAIValidationFinding:
    path: str
    line: int
    rule: str
    severity: str
    message: str
    function: str | None = None
    column: str | None = None


@dataclass
class FreqAIValidationReport:
    ok: bool
    files_checked: int
    findings: list[FreqAIValidationFinding]
    feature_columns: list[FreqAIColumnReference]
    target_columns: list[FreqAIColumnReference]
    allowed_target_shift_lines: list[dict[str, Any]]
    label_notice: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "files_checked": self.files_checked,
            "findings": [asdict(finding) for finding in self.findings],
            "feature_columns": [asdict(column) for column in self.feature_columns],
            "target_columns": [asdict(column) for column in self.target_columns],
            "allowed_target_shift_lines": self.allowed_target_shift_lines,
            "label_notice": self.label_notice,
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


DEFAULT_FREQAI_DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("lightgbm"),
    DependencySpec("xgboost"),
    DependencySpec("tensorboard"),
    DependencySpec("datasieve"),
)


def check_freqai_dependencies(
    dependencies: Sequence[DependencySpec] = DEFAULT_FREQAI_DEPENDENCIES,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    package_version: Callable[[str], str] = metadata.version,
) -> FreqAIEnvironmentReport:
    statuses = [
        _check_dependency(
            dependency,
            import_module=import_module,
            package_version=package_version,
        )
        for dependency in dependencies
    ]
    ok = not any(status.required and not status.installed for status in statuses)
    return FreqAIEnvironmentReport(
        ok=ok,
        dependencies=statuses,
        python_version=sys.version.split()[0],
        generated_at=datetime.now(UTC).isoformat(),
    )


def missing_required_dependencies(report: FreqAIEnvironmentReport) -> list[str]:
    return [
        dependency.import_name
        for dependency in report.dependencies
        if dependency.required and not dependency.installed
    ]


def validate_freqai_strategy_paths(paths: Sequence[Path]) -> FreqAIValidationReport:
    files = _iter_python_files(paths)
    findings: list[FreqAIValidationFinding] = []
    feature_columns: list[FreqAIColumnReference] = []
    target_columns: list[FreqAIColumnReference] = []
    allowed_target_shift_lines: list[dict[str, Any]] = []

    for file_path in files:
        file_report = validate_freqai_strategy_file(file_path)
        findings.extend(file_report.findings)
        feature_columns.extend(file_report.feature_columns)
        target_columns.extend(file_report.target_columns)
        allowed_target_shift_lines.extend(file_report.allowed_target_shift_lines)

    ok = not any(finding.severity == "error" for finding in findings)
    return FreqAIValidationReport(
        ok=ok,
        files_checked=len(files),
        findings=findings,
        feature_columns=feature_columns,
        target_columns=target_columns,
        allowed_target_shift_lines=allowed_target_shift_lines,
        label_notice=FREQAI_LABEL_NOTICE,
        generated_at=datetime.now(UTC).isoformat(),
    )


def validate_freqai_strategy_file(path: Path) -> FreqAIValidationReport:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        finding = FreqAIValidationFinding(
            path=str(path),
            line=exc.lineno or 0,
            rule="syntax_error",
            severity="error",
            message=str(exc),
        )
        return FreqAIValidationReport(
            ok=False,
            files_checked=1,
            findings=[finding],
            feature_columns=[],
            target_columns=[],
            allowed_target_shift_lines=[],
            label_notice=FREQAI_LABEL_NOTICE,
            generated_at=datetime.now(UTC).isoformat(),
        )

    visitor = _FreqAIValidationVisitor(path)
    visitor.visit(tree)
    visitor.finalize()
    ok = not any(finding.severity == "error" for finding in visitor.findings)
    return FreqAIValidationReport(
        ok=ok,
        files_checked=1,
        findings=visitor.findings,
        feature_columns=visitor.feature_columns,
        target_columns=visitor.target_columns,
        allowed_target_shift_lines=visitor.allowed_target_shift_lines,
        label_notice=FREQAI_LABEL_NOTICE,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _check_dependency(
    dependency: DependencySpec,
    *,
    import_module: Callable[[str], Any],
    package_version: Callable[[str], str],
) -> DependencyStatus:
    package_name = dependency.package_name or dependency.import_name
    try:
        module = import_module(dependency.import_name)
    except Exception as exc:
        return DependencyStatus(
            name=dependency.import_name,
            import_name=dependency.import_name,
            package_name=package_name,
            required=dependency.required,
            installed=False,
            version=None,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    return DependencyStatus(
        name=dependency.import_name,
        import_name=dependency.import_name,
        package_name=package_name,
        required=dependency.required,
        installed=True,
        version=_dependency_version(module, package_name, package_version),
        error=None,
    )


def _dependency_version(
    module: Any,
    package_name: str,
    package_version: Callable[[str], str],
) -> str | None:
    try:
        return package_version(package_name)
    except metadata.PackageNotFoundError:
        module_version = getattr(module, "__version__", None)
        return str(module_version) if module_version is not None else None


class _FreqAIValidationVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[FreqAIValidationFinding] = []
        self.feature_columns: list[FreqAIColumnReference] = []
        self.target_columns: list[FreqAIColumnReference] = []
        self.allowed_target_shift_lines: list[dict[str, Any]] = []
        self.freqai_start_lines: list[int] = []
        self.freqai_method_lines: list[int] = []
        self._function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.name == "set_freqai_targets" or node.name.startswith("feature_engineering_"):
            self.freqai_method_lines.append(getattr(node, "lineno", 0))
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        if node.name == "set_freqai_targets" or node.name.startswith("feature_engineering_"):
            self.freqai_method_lines.append(getattr(node, "lineno", 0))
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            self._inspect_assignment_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._inspect_assignment_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._inspect_assignment_target(node.target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if self._attr_name(node.func) == "shift" and self._has_negative_shift_arg(node):
            self._inspect_negative_shift(node)
        if self._is_freqai_start_call(node):
            self.freqai_start_lines.append(getattr(node, "lineno", 0))
        self.generic_visit(node)

    def finalize(self) -> None:
        if self.freqai_method_lines and not self.freqai_start_lines:
            self.findings.append(
                FreqAIValidationFinding(
                    path=str(self.path),
                    line=self.freqai_method_lines[0],
                    rule="freqai_start_required",
                    severity="error",
                    message=(
                        "Strategies defining FreqAI feature or target methods must call "
                        "self.freqai.start(dataframe, metadata, self) so predictions are generated."
                    ),
                    function=None,
                    column=None,
                )
            )

    def _inspect_assignment_target(self, target: ast.AST) -> None:
        function = self._current_function()
        if not function:
            return

        for column in self._dataframe_columns(target):
            if function.startswith("feature_engineering_"):
                self.feature_columns.append(self._column_reference(target, function, column))
                if not column.startswith("%"):
                    self._add(
                        target,
                        "freqai_feature_prefix",
                        "error",
                        "FreqAI feature columns created in feature_engineering_* must start with '%'.",
                        column=column,
                    )
            elif function == "set_freqai_targets":
                self.target_columns.append(self._column_reference(target, function, column))
                if not self._is_valid_target_prefix(column):
                    self._add(
                        target,
                        "freqai_target_prefix",
                        "error",
                        "FreqAI target/label columns created in set_freqai_targets must start with '&'.",
                        column=column,
                    )

    def _inspect_negative_shift(self, node: ast.Call) -> None:
        function = self._current_function()
        if function == "set_freqai_targets":
            self.allowed_target_shift_lines.append(
                {
                    "path": str(self.path),
                    "line": getattr(node, "lineno", 0),
                    "function": function,
                    "reason": "supervised target generation",
                }
            )
            return

        if self._in_signal_or_feature_context():
            self._add(
                node,
                "freqai_shift_outside_targets",
                "error",
                "Negative shift is allowed only inside set_freqai_targets target generation.",
            )
        else:
            self._add(
                node,
                "freqai_shift_review",
                "warning",
                "Negative shift found outside set_freqai_targets; review for lookahead risk.",
            )

    def _dataframe_columns(self, node: ast.AST) -> list[str]:
        if isinstance(node, ast.Subscript):
            direct_column = self._direct_dataframe_column(node)
            if direct_column:
                return [direct_column]
            return self._loc_dataframe_columns(node)
        if isinstance(node, ast.Tuple):
            columns: list[str] = []
            for element in node.elts:
                columns.extend(self._dataframe_columns(element))
            return columns
        return []

    def _direct_dataframe_column(self, node: ast.Subscript) -> str | None:
        if isinstance(node.value, ast.Name) and node.value.id == "dataframe":
            return self._literal_string(node.slice)
        return None

    def _loc_dataframe_columns(self, node: ast.Subscript) -> list[str]:
        value = node.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "loc"
            and isinstance(value.value, ast.Name)
            and value.value.id == "dataframe"
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) >= 2
        ):
            return []
        return self._literal_strings(node.slice.elts[1])

    def _literal_string(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _literal_strings(self, node: ast.AST) -> list[str]:
        literal = self._literal_string(node)
        if literal:
            return [literal]
        if isinstance(node, (ast.List, ast.Tuple)):
            values: list[str] = []
            for element in node.elts:
                element_literal = self._literal_string(element)
                if element_literal:
                    values.append(element_literal)
            return values
        return []

    def _has_negative_shift_arg(self, node: ast.Call) -> bool:
        if node.args and self._is_negative_expression(node.args[0]):
            return True
        return any(
            keyword.arg == "periods" and self._is_negative_expression(keyword.value)
            for keyword in node.keywords
        )

    def _is_negative_expression(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value < 0
        return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)

    def _in_signal_or_feature_context(self) -> bool:
        function = self._current_function()
        if not function:
            return False
        return function.startswith("populate_") or function.startswith("feature_engineering_")

    def _current_function(self) -> str | None:
        return self._function_stack[-1] if self._function_stack else None

    def _is_valid_target_prefix(self, column: str) -> bool:
        return column.startswith("&")

    def _column_reference(
        self, node: ast.AST, function: str, column: str
    ) -> FreqAIColumnReference:
        return FreqAIColumnReference(
            path=str(self.path),
            line=getattr(node, "lineno", 0),
            function=function,
            column=column,
        )

    def _attr_name(self, node: ast.AST) -> str:
        return node.attr if isinstance(node, ast.Attribute) else ""

    def _is_freqai_start_call(self, node: ast.Call) -> bool:
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "start"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "freqai"
        )

    def _add(
        self,
        node: ast.AST,
        rule: str,
        severity: str,
        message: str,
        *,
        column: str | None = None,
    ) -> None:
        self.findings.append(
            FreqAIValidationFinding(
                path=str(self.path),
                line=getattr(node, "lineno", 0),
                rule=rule,
                severity=severity,
                message=message,
                function=self._current_function(),
                column=column,
            )
        )


def _iter_python_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts))
    return files
