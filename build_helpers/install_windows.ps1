# Downloads don't work automatically, since the URL is regenerated via javascript.
# Downloaded from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

python -m pip install --upgrade pip

$pyv = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($pyv -eq '3.7') {
    pip install build_helpers\TA_Lib-0.4.21-cp37-cp37m-win_amd64.whl
}
if ($pyv -eq '3.8') {
    pip install build_helpers\TA_Lib-0.4.21-cp38-cp38-win_amd64.whl
}
if ($pyv -eq '3.9') {
    pip install build_helpers\TA_Lib-0.4.21-cp39-cp39-win_amd64.whl
}

pip install -r requirements-dev.txt
pip install -e .
