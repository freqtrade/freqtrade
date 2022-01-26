class Config:
    BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH = ""
    BACKTEST_YEAR = 2020
    BACKTEST_MONTH_INDEX = 9
    IS_BACKTEST = True
    WORKSPACE_PATH = "workspace2" if IS_BACKTEST else "workspace"
    EXECUTION_PATH = "/root/" + WORKSPACE_PATH + "/execution/"
    IS_PARRALER_EXECUTION = False
