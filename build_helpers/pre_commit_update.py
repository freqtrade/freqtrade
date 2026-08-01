# File used in CI to ensure pre-commit dependencies are kept up-to-date.

import argparse
import re
import sys
from pathlib import Path

import yaml


pre_commit_file = Path(".pre-commit-config.yaml")
require_dev = Path("requirements-dev.txt")
require = Path("requirements.txt")


parser = argparse.ArgumentParser()
parser.add_argument("--update", action="store_true")
args = parser.parse_args()


def replace_dependency_version(pre_commit_text: str, dependency: str) -> tuple[str, bool]:
    """
    Regex-based replacement of a dependency version in the pre-commit config file.
    using regex here ensures we only replace the version of the dependency while
    keeping the overall file intact.
    """
    package_name = dependency.split("==", 1)[0]
    pattern = re.compile(rf"^(\s*-\s+){re.escape(package_name)}==.*$", re.MULTILINE)
    updated_text, replacements = pattern.subn(rf"\1{dependency}", pre_commit_text, count=1)
    return updated_text, replacements > 0 and updated_text != pre_commit_text


with require_dev.open("r") as rfile:
    requirements = rfile.readlines()

with require.open("r") as rfile:
    requirements.extend(rfile.readlines())

# Extract relevant types only
supported = ("types-", "SQLAlchemy", "scipy-stubs")

# Find relevant dependencies
# Only keep the first part of the line up to the first space
type_reqs = [r.strip("\n").split()[0] for r in requirements if r.startswith(supported)]

with pre_commit_file.open("r") as file:
    pre_commit_text = file.read()

updated = False
for req in type_reqs:
    pre_commit_text, req_updated = replace_dependency_version(pre_commit_text, req)
    updated = updated or req_updated

if args.update and updated:
    with pre_commit_file.open("w") as file:
        file.write(pre_commit_text)

with pre_commit_file.open("r") as file:
    f = yaml.load(file, Loader=yaml.SafeLoader)


mypy_repo = [
    repo for repo in f["repos"] if repo["repo"] == "https://github.com/pre-commit/mirrors-mypy"
]

hooks = mypy_repo[0]["hooks"][0]["additional_dependencies"]

errors = []
for hook in hooks:
    if hook not in type_reqs:
        errors.append(f"{hook} is missing in requirements-dev.txt.")

for req in type_reqs:
    if req not in hooks:
        errors.append(f"{req} is missing in pre-commit config file.")

if updated:
    if args.update:
        errors.append(".pre-commit-config.yaml was updated to match the requirements files.")
    else:
        errors.append(
            ".pre-commit-config.yaml is outdated. Run build_helpers/pre_commit_update.py --update."
        )


if errors:
    for e in errors:
        print(e)
    sys.exit(1 if not (args.update and updated) else 0)

sys.exit(0)
