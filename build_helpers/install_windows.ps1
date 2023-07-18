# Downloads don't work automatically, since the URL is regenerated via javascript.
# Downloaded from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

python -m pip install --upgrade pip wheel

$pyv = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"


pip install --user build_helpers\TA_Lib-*.whl

pip install -r requirements-dev.txt
pip install -e .
