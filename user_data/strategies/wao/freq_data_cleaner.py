import json
import datetime
import time
import os
import sys
from brain_config import BrainConfig


def clean_json():
    print("clean_json: json_path = " + BrainConfig.BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)
    file = open(BrainConfig.BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)
    list = []
    data = json.load(file)
    for datas in data:
        unix = datas[0] / 1000
        unix = int(unix)
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(unix)))
        date = datetime.datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")
        year = date.year
        month = date.month
        if year == int(BrainConfig.BACKTEST_DATA_CLEANER_YEAR) and month == int(
                BrainConfig.BACKTEST_DATA_CLEANER_MONTH_INDEX) + 1:
            list.append(datas)
    json_object = json.dumps(list)
    file.close()
    write_to_json(json_object)


def write_to_json(json_object):
    print("write_to_json: json_path = " + BrainConfig.BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)
    with open("temp.json", "w") as outfile:
        outfile.write(json_object)
    os.rename("temp.json", BrainConfig.BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)


if len(sys.argv) < 2:
    exit("""Incorrect number of arguments. 
    python3 freq_data_cleaner.py [json_file] 
    """)
else:
    BrainConfig.BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH = sys.argv[1]
    clean_json()
