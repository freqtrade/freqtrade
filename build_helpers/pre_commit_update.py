import argparse
import sys
from pathlib import Path

import yaml


PRE_COMMIT_FILE = Path(".pre-commit-config.yaml")
REQUIRE_DEV = Path("requirements-dev.txt")
REQUIRE = Path("requirements.txt")
SUPPORTED = ("types-", "SQLAlchemy", "scipy-stubs")
MYPY_REPO = "https://github.com/pre-commit/mirrors-mypy"


def get_type_requirements(requirement_paths: list[Path]) -> list[str]:
    requirements: list[str] = []
    for path in requirement_paths:
        with path.open("r") as rfile:
            requirements.extend(rfile.readlines())

    return [line.strip("\n").split()[0] for line in requirements if line.startswith(SUPPORTED)]


def get_mypy_dependencies(pre_commit_file: Path) -> list[str]:
    with pre_commit_file.open("r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    mypy_repo = [repo for repo in config["repos"] if repo["repo"] == MYPY_REPO]
    return mypy_repo[0]["hooks"][0]["additional_dependencies"]


def get_dependency_errors(type_reqs: list[str], hook_deps: list[str]) -> list[str]:
    errors = []
    for hook in hook_deps:
        if hook not in type_reqs:
            errors.append(f"{hook} is missing in requirements-dev.txt.")

    for req in type_reqs:
        if req not in hook_deps:
            errors.append(f"{req} is missing in pre-config file.")

    return errors


def replace_mypy_additional_dependencies(pre_commit_file: Path, dependencies: list[str]) -> None:
    lines = pre_commit_file.read_text().splitlines(keepends=True)

    repo_index = next(i for i, line in enumerate(lines) if MYPY_REPO in line)
    additional_index = next(
        i
        for i in range(repo_index, len(lines))
        if lines[i].lstrip().startswith("additional_dependencies:")
    )

    additional_indent = len(lines[additional_index]) - len(lines[additional_index].lstrip())
    item_indent = " " * (additional_indent + 2)

    end_index = additional_index + 1
    while end_index < len(lines):
        line = lines[end_index]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped and indent <= additional_indent:
            break
        end_index += 1

    replacement = [lines[additional_index]] + [
        f"{item_indent}- {dependency}\n" for dependency in dependencies
    ]
    pre_commit_file.write_text("".join(lines[:additional_index] + replacement + lines[end_index:]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or sync mypy pre-commit dependencies.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Update mypy additional_dependencies in .pre-commit-config.yaml",
    )
    args = parser.parse_args()

    type_reqs = get_type_requirements([REQUIRE_DEV, REQUIRE])
    hook_deps = get_mypy_dependencies(PRE_COMMIT_FILE)
    errors = get_dependency_errors(type_reqs, hook_deps)

    if errors and args.fix:
        replace_mypy_additional_dependencies(PRE_COMMIT_FILE, type_reqs)
        hook_deps = get_mypy_dependencies(PRE_COMMIT_FILE)
        errors = get_dependency_errors(type_reqs, hook_deps)

    if errors:
        for error in errors:
            print(error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
