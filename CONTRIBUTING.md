# Contribute to freqtrade

Feel like our bot is missing a feature? We welcome your pull requests! Few pointers for contributions:

- Create your PR against the `develop` branch, not `master`.
- New features need to contain unit tests and must be PEP8 
conformant (max-line-length = 100).

If you are unsure, discuss the feature on our [Slack](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE)
or in a [issue](https://github.com/gcarq/freqtrade/issues) before a PR.


**Before sending the PR:**

## 1. Run unit tests

All unit tests must pass. If a unit test is broken, change your code to 
make it pass. It means you have introduced a regression.

**Test the whole project**
```bash
pytest freqtrade
```

**Test only one file**
```bash
pytest freqtrade/tests/test_<file_name>.py
```

**Test only one method from one file**
```bash
pytest freqtrade/tests/test_<file_name>.py::test_<method_name>
```

## 2. Test if your code is PEP8 compliant
**Install packages** (If not already installed)
```bash
pip3.6 install flake8 coveralls
``` 
**Run Flake8**
```bash
flake8 freqtrade
```


