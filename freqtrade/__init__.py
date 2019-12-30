""" FreqTrade bot """
__version__ = 'develop'

if __version__ == 'develop':

    try:
        import subprocess
        __version__ = 'develop-' + subprocess.check_output(
            ['git', 'log', '--format="%h"', '-n 1'],
            stderr=subprocess.DEVNULL).decode("utf-8").rstrip().strip('"')
    except Exception:
        # git not available, ignore
        pass
