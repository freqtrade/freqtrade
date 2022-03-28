class BrainConfig:

    BACKTEST_DUP = '0.45'
    BACKTEST_MAX_COUNT_DUP = '3'
    BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH = ""
    BACKTEST_COIN = 'ETH'
    BACKTEST_MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    BACKTEST_DATA_CLEANER_YEAR = 2021
    BACKTEST_DATA_CLEANER_MONTH_INDEX = 4
    IS_BACKTEST = False
    WORKSPACE_PATH = "workspace2" if IS_BACKTEST else "workspace"
    _429_DIRECTORY = "/root/" + WORKSPACE_PATH + "/freqtrade/_429_directory/"
    IS_PARALLEL_EXECUTION = False
    BACKTEST_THROTTLE_SECOND = 1
