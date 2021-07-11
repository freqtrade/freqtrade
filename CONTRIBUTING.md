# Contributing

## Contribute to freqtrade

Feel like our bot is missing a feature? We welcome your pull requests! 

Issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) can be good first contributions, and will help get you familiar with the codebase.

Few pointers for contributions:

- Create your PR against the `develop` branch, not `stable`.
- New features need to contain unit tests, must conform to PEP8 (max-line-length = 100) and should be documented with the introduction PR.
- PR's can be declared as `[WIP]` - which signify Work in Progress Pull Requests (which are not finished).

If you are unsure, discuss the feature on our [discord server](https://discord.gg/p7nuUNVfP7) or in a [issue](https://github.com/freqtrade/freqtrade/issues) before a Pull Request.

## Getting started

Best start by reading the [documentation](https://www.freqtrade.io/) to get a feel for what is possible with the bot, or head straight to the [Developer-documentation](https://www.freqtrade.io/en/latest/developer/) (WIP) which should help you getting started.

## Before sending the PR

### 1. Run unit tests

All unit tests must pass. If a unit test is broken, change your code to 
make it pass. It means you have introduced a regression.

#### Test the whole project

```bash
pytest
```

#### Test only one file

```bash
pytest tests/test_<file_name>.py
```

#### Test only one method from one file

```bash
pytest tests/test_<file_name>.py::test_<method_name>
```

### 2. Test if your code is PEP8 compliant

#### Run Flake8

```bash
flake8 freqtrade tests scripts
```

We receive a lot of code that fails the `flake8` checks.
To help with that, we encourage you to install the git pre-commit 
hook that will warn you when you try to commit code that fails these checks. 
Guide for installing them is [here](http://flake8.pycqa.org/en/latest/user/using-hooks.html).

### 3. Test if all type-hints are correct

#### Run mypy

``` bash
mypy freqtrade
```

### 4. Ensure all imports are correct

#### Run isort

``` bash
isort .
```

## (Core)-Committer Guide

### Process: Pull Requests

How to prioritize pull requests, from most to least important:

1. Fixes for broken tests. Broken means broken on any supported platform or Python version.
1. Extra tests to cover corner cases.
1. Minor edits to docs.
1. Bug fixes.
1. Major edits to docs.
1. Features.

Ensure that each pull request meets all requirements in the Contributing document.

### Process: Issues

If an issue is a bug that needs an urgent fix, mark it for the next patch release.
Then either fix it or mark as please-help.

For other issues: encourage friendly discussion, moderate debate, offer your thoughts.

### Process: Your own code changes

All code changes, regardless of who does them, need to be reviewed and merged by someone else.
This rule applies to all the core committers.

Exceptions:

- Minor corrections and fixes to pull requests submitted by others.
- While making a formal release, the release manager can make necessary, appropriate changes.
- Small documentation changes that reinforce existing subject matter. Most commonly being, but not limited to spelling and grammar corrections.

### Responsibilities

- Ensure cross-platform compatibility for every change that's accepted. Windows, Mac & Linux.
- Ensure no malicious code is introduced into the core code.
- Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
- Keep feature versions as small as possible, preferably one new feature per version.
- Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See the Python Community Code of Conduct (https://www.python.org/psf/codeofconduct/).

### Becoming a Committer

Contributors may be given commit privileges. Preference will be given to those with:

1. Past contributions to Freqtrade and other related open-source projects. Contributions to Freqtrade include both code (both accepted and pending) and friendly participation in the issue tracker and Pull request reviews. Quantity and quality are considered.
1. A coding style that the other core committers find simple, minimal, and clean.
1. Access to resources for cross-platform development and testing.
1. Time to devote to the project regularly.

Being a Committer does not grant write permission on `develop` or `stable` for security reasons (Users trust Freqtrade with their Exchange API keys).

After being Committer for some time, a Committer may be named Core Committer and given full repository access.
