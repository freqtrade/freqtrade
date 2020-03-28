""" Freqtrade bot """
__version__ = '2020.3'

if __version__ == 'develop':

    try:
        import subprocess

        __version__ = 'develop-' + subprocess.check_output(
            ['git', 'log', '--format="%h"', '-n 1'],
            stderr=subprocess.DEVNULL).decode("utf-8").rstrip().strip('"')

        # from datetime import datetime
        # last_release = subprocess.check_output(
        #     ['git', 'tag']
        # ).decode('utf-8').split()[-1].split(".")
        # # Releases are in the format "2020.1" - we increment the latest version for dev.
        # prefix = f"{last_release[0]}.{int(last_release[1]) + 1}"
        # dev_version = int(datetime.now().timestamp() // 1000)
        # __version__ = f"{prefix}.dev{dev_version}"

        #  subprocess.check_output(
        #     ['git', 'log', '--format="%h"', '-n 1'],
        #     stderr=subprocess.DEVNULL).decode("utf-8").rstrip().strip('"')
    except Exception:
        # git not available, ignore
        pass
