from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from typing import Any, Callable, Sequence


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
