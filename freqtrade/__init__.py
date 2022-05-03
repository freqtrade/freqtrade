""" Freqtrade bot """
__version__ = '2022.4.2'

if 'dev' in __version__:
    try:
        import subprocess

        __version__ = __version__ + '-' + subprocess.check_output(
            ['git', 'log', '--format="%h"', '-n 1'],
            stderr=subprocess.DEVNULL).decode("utf-8").rstrip().strip('"')

    except Exception:  # pragma: no cover
        # git not available, ignore
        try:
            # Try Fallback to freqtrade_commit file (created by CI while building docker image)
            from pathlib import Path
            versionfile = Path('./freqtrade_commit')
            if versionfile.is_file():
                __version__ = f"docker-{versionfile.read_text()[:8]}"
        except Exception:
            pass
