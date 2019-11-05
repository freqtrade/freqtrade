Invoke-WebRequest -Uri "https://download.lfd.uci.edu/pythonlibs/g5apjq5m/TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl" -OutFile "TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl"

pip install TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl

pip install -r requirements-dev.txt
pip install -e .
