# Downloads don't work automatically, since the URL is regenerated via javascript.
# Downloaded from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Invoke-WebRequest -Uri "https://download.lfd.uci.edu/pythonlibs/xxxxxxx/TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl" -OutFile "TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl"

python -m pip install --upgrade pip==21.0.0

$pyv = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($pyv -eq '3.7') {
    pip install build_helpers\TA_Lib-0.4.19-cp37-cp37m-win_amd64.whl
}
if ($pyv -eq '3.8') {
    pip install build_helpers\TA_Lib-0.4.19-cp38-cp38-win_amd64.whl
}

pip install -r requirements-dev.txt
pip install -e .
