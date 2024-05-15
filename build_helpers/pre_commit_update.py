# File used in CI to ensure pre-commit dependencies are kept up-to-date.

import sys
from pathlib import Path

import yaml


pre_commit_file = Path(".pre-commit-config.yaml")
require_dev = Path("requirements-dev.txt")
require = Path("requirements.txt")

with require_dev.open("r") as rfile:
    requirements = rfile.readlines()

with require.open("r") as rfile:
    requirements.extend(rfile.readlines())

# Extract types only
type_reqs = [
    r.strip("\n") for r in requirements if r.startswith("types-") or r.startswith("SQLAlchemy")
]

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
        errors.append(f"{req} is missing in pre-config file.")


if errors:
    for e in errors:
        print(e)
    sys.exit(1)

sys.exit(0)
