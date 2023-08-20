# vendored Wheels compiled via https://github.com/xmatthias/ta-lib-python/tree/ta_bundled_040

python -m pip install --upgrade pip wheel

$pyv = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"


pip install --find-links=build_helpers\ --prefer-binary  TA-Lib

pip install -r requirements-dev.txt
pip install -e .
