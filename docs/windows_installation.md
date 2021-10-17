# Windows installation

We **strongly** recommend that Windows users use [Docker](docker_quickstart.md) as this will work much easier and smoother (also more secure).

If that is not possible, try using the Windows Linux subsystem (WSL) - for which the Ubuntu instructions should work.
Otherwise, try the instructions below.

## Install freqtrade manually

!!! Note
    Make sure to use 64bit Windows and 64bit Python to avoid problems with backtesting or hyperopt due to the memory constraints 32bit applications have under Windows.

!!! Hint
    Using the [Anaconda Distribution](https://www.anaconda.com/distribution/) under Windows can greatly help with installation problems. Check out the [Anaconda installation section](installation.md#Anaconda) in this document for more information.

### 1. Clone the git repository

```bash
git clone https://github.com/freqtrade/freqtrade.git
```

### 2. Install ta-lib

Install ta-lib according to the [ta-lib documentation](https://github.com/mrjbq7/ta-lib#windows).

As compiling from source on windows has heavy dependencies (requires a partial visual studio installation), there is also a repository of unofficial pre-compiled windows Wheels [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib), which need to be downloaded and installed using `pip install TA_Lib-0.4.21-cp38-cp38-win_amd64.whl` (make sure to use the version matching your python version).

Freqtrade provides these dependencies for the latest 2 Python versions (3.7 and 3.8) and for 64bit Windows.
Other versions must be downloaded from the above link.

``` powershell
cd \path\freqtrade
python -m venv .env
.env\Scripts\activate.ps1
# optionally install ta-lib from wheel
# Eventually adjust the below filename to match the downloaded wheel
pip install build_helpers/TA_Lib-0.4.19-cp38-cp38-win_amd64.whl
pip install -r requirements.txt
pip install -e .
freqtrade
```

!!! Note "Use Powershell"
    The above installation script assumes you're using powershell on a 64bit windows.
    Commands for the legacy CMD windows console may differ.

> Thanks [Owdr](https://github.com/Owdr) for the commands. Source: [Issue #222](https://github.com/freqtrade/freqtrade/issues/222)

### Error during installation on Windows

``` bash
error: Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools": http://landinghub.visualstudio.com/visual-cpp-build-tools
```

Unfortunately, many packages requiring compilation don't provide a pre-built wheel. It is therefore mandatory to have a C/C++ compiler installed and available for your python environment to use.

The easiest way is to download install Microsoft Visual Studio Community [here](https://visualstudio.microsoft.com/downloads/) and make sure to install "Common Tools for Visual C++" to enable building C code on Windows. Unfortunately, this is a heavy download / dependency (~4Gb) so you might want to consider WSL or [docker compose](docker_quickstart.md) first.

---
