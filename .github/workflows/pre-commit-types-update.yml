name: Pre-commit Types update

on:
  pull_request:
    branches:
      - "develop"

concurrency:
  group: "${{ github.workflow }}-${{ github.ref }}-${{ github.event_name }}"
  cancel-in-progress: true
permissions: {}

jobs:
  mypy-version-update:
    name: "Pre-commit mypy type versions update"
    runs-on: ubuntu-24.04
    # Only run this job for pull requests created by dependabot[bot]
    if: >
      github.event.pull_request.user.login == 'dependabot[bot]' &&
      github.repository == github.event.pull_request.head.repo.full_name &&
      github.event_name == 'pull_request' &&
      startsWith(github.head_ref, 'dependabot/')

    environment:
      name: dependabot-pulls

    steps:
    - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      with:
        persist-credentials: true
        token: ${{ secrets.REPO_SCOPED_TOKEN_DEP }}
        ref: ${{ github.head_ref || github.ref }}

    - name: Install uv and Python 🐍
      uses: astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990 # v8.3.2
      with:
        activate-environment: true
        python-version: "3.13"

    - name: Install PyYAML
      run: |

    - name: pre-commit dependencies
      run: |
        uv pip install $(grep -E "^pyyaml==" requirements-dev.txt)
        python build_helpers/pre_commit_update.py --update

    - uses: stefanzweifel/git-auto-commit-action@4a55954c782fc1ea30b9056cd3e7a2b40ca8887d # v7.2.0
      with:
        commit_message: "chore(deps): Apply pre-commit types update"
        commit_user_name: Freqtrade Bot
        commit_user_email: 154552126+freqtrade-bot@users.noreply.github.com
        commit_author: Freqtrade Bot <154552126+freqtrade-bot@users.noreply.github.com>
