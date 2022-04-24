class BrainConfig:

    BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH = ""
    BACKTEST_MONTH_LIST = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    BACKTEST_DATA_CLEANER_YEAR = 2021
    BACKTEST_DATA_CLEANER_MONTH_INDEX = 4
    IS_BACKTEST = False
    WORKSPACE = "/workspace2" if IS_BACKTEST else "/workspace"
    ROOT = "/root"
    EXECUTION_PATH = ROOT + WORKSPACE + "/execution"
    FREQTRADE_PATH = ROOT + WORKSPACE + "/freqtrade"
    _429_DIRECTORY = FREQTRADE_PATH + "/_429_directory/"
    BACKTEST_SIGNAL_LIST_PICKLE_FILE_PATH = FREQTRADE_PATH + "/_backtest_list.pickle"
    BACKTEST_SIGNAL_LIST = []
    BACKTEST_THROTTLE_SECOND = 1
    MODE = "test" # test or prod
    CUMULATIVE_PROFIT_FILE_PATH = EXECUTION_PATH + "/_cumulative_profit.txt"
