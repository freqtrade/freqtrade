import sys
import threading
import watchdog
import os
import time
import datetime
from wao.brain_config import BrainConfig
from wao._429_watcher import _429_Watcher

sys.path.append(BrainConfig.EXECUTION_PATH)
from config import Config
from romeo import Romeo, RomeoExitPriceType



#1. read from BACKTEST_TABLE_FILE_PATH to a map
#2. start romeo with sending the map and setting is_backtest to true



