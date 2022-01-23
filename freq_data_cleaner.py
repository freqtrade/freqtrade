import json
import datetime
import time
import os
import sys
from user_data.strategies.util import BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH, BACKTEST_YEAR, BACKTEST_MONTH

def clean_json():
    print("clean_json: json_path = " + BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)
    file = open(BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)
    list = []
    data = json.load(file)
    for datas in data:
        unix = datas[0]/1000
        unix = int(unix)
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(unix)))
        date = datetime.datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")
        year = date.year
        month = date.month
        if year == int(BACKTEST_YEAR) and month == int(BACKTEST_MONTH):
            list.append(datas)
    json_object = json.dumps(list)
    file.close()
    write_to_json(json_object)

def write_to_json(json_object):  
    print("write_to_json: json_path = " + BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)  
    with open("temp.json", "w") as outfile:
        outfile.write(json_object)
    os.rename("temp.json", BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)

if len(sys.argv) < 4:
    exit("""Incorrect number of arguments. 
    python3 freq_data_cleaner.py [json_file] [month] [year]
    """)
else:
    BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH = sys.argv[1]
    BACKTEST_MONTH = sys.argv[2]
    BACKTEST_YEAR = sys.argv[3]
    clean_json()