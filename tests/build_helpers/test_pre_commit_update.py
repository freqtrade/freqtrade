from pathlib import Path

from build_helpers.pre_commit_update import (
    get_dependency_errors,
    get_mypy_dependencies,
    get_type_requirements,
    replace_mypy_additional_dependencies,
)


def write_pre_commit(path: Path, dependencies: list[str]) -> None:
    dependency_lines = "\n".join(f"          - {dependency}" for dependency in dependencies)
    path.write_text(
        f"""
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        exclude: build_helpers
        additional_dependencies:
{dependency_lines}
        # stages: [push]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
""".lstrip()
    )


def test_get_dependency_errors_reports_both_directions() -> None:
    type_reqs = ["types-requests==1", "SQLAlchemy==2"]
    hook_deps = ["types-requests==0", "scipy-stubs==1"]

    assert get_dependency_errors(type_reqs, hook_deps) == [
        "types-requests==0 is missing in requirements-dev.txt.",
        "scipy-stubs==1 is missing in requirements-dev.txt.",
        "types-requests==1 is missing in pre-config file.",
        "SQLAlchemy==2 is missing in pre-config file.",
    ]


def test_replace_mypy_additional_dependencies_syncs_only_mypy_block(tmp_path: Path) -> None:
    pre_commit_file = tmp_path / ".pre-commit-config.yaml"
    write_pre_commit(pre_commit_file, ["types-requests==0", "scipy-stubs==1"])

    replace_mypy_additional_dependencies(
        pre_commit_file,
        ["types-requests==1", "SQLAlchemy==2"],
    )

    assert get_mypy_dependencies(pre_commit_file) == ["types-requests==1", "SQLAlchemy==2"]
    assert "repo: https://github.com/pre-commit/pre-commit-hooks" in pre_commit_file.read_text()


def test_get_type_requirements_reads_supported_lines_in_order(tmp_path: Path) -> None:
    requirements_dev = tmp_path / "requirements-dev.txt"
    requirements = tmp_path / "requirements.txt"
    requirements_dev.write_text(
        "\n".join(
            [
                "ruff==1",
                "types-cachetools==1  # comment",
                "scipy-stubs==2",
            ]
        )
    )
    requirements.write_text("SQLAlchemy==2\nrequests==1\n")

    assert get_type_requirements([requirements_dev, requirements]) == [
        "types-cachetools==1",
        "scipy-stubs==2",
        "SQLAlchemy==2",
    ]
