class BrainConfig:

    BACKTEST_DUP = '0.45'
    BACKTEST_MAX_COUNT_DUP = '3'
    BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH = ""
    BACKTEST_COIN = 'ETH'
    BACKTEST_MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    BACKTEST_DATA_CLEANER_YEAR = 2021
    BACKTEST_DATA_CLEANER_MONTH_INDEX = 4
    IS_BACKTEST = False
    CHOP_TESTER_WORKSPACE = "workspace2"
    WORKSPACE_PATH = "workspace2" if IS_BACKTEST else "workspace"
    ROOT_DIRECTORY = "/root/"
    EXECUTION_PATH = ROOT_DIRECTORY + "workspace/execution"
    _429_DIRECTORY = ROOT_DIRECTORY + WORKSPACE_PATH + "/freqtrade/_429_directory/"
    BACKTEST_EXECUTION_LIST_FILE = ROOT_DIRECTORY + WORKSPACE_PATH + "/freqtrade/_backtest_list.txt"
    BACKTEST_EXECUTION_LIST = []
    IS_PARALLEL_EXECUTION = True
    BACKTEST_THROTTLE_SECOND = 1
    MODE = "test" # test or prod
    CUMULATIVE_PROFIT_FILE_PATH = ROOT_DIRECTORY + WORKSPACE_PATH + "/execution/_cumulative_profit.txt"
