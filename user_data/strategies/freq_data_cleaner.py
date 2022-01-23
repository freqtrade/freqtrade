import json
import datetime
import time
import os
import sys
from util import BACKTEST_JSON_PATH, BACKTEST_YEAR, BACKTEST_MONTH

def clean_json():
    print("clean_json: json_path = " + BACKTEST_JSON_PATH)
    file = open(BACKTEST_JSON_PATH)
    list = []
    data = json.load(file)
    for datas in data:
        datas[0] = datas[0]/1000
        datas[0] = int(datas[0])
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(datas[0])))
        date = datetime.datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")
        year = date.year
        month = date.month
        if year == int(BACKTEST_YEAR) and month == int(BACKTEST_MONTH):
            list.append(datas)
    json_object = json.dumps(list)
    file.close()
    write_to_json(json_object)

def write_to_json(json_object):  
    print("write_to_json: json_path = " + BACKTEST_JSON_PATH)  
    with open("temp.json", "w") as outfile:
        outfile.write(json_object)
    os.rename("temp.json", BACKTEST_JSON_PATH)

if len(sys.argv) < 4:
    exit("""Incorrect number of arguments. 
    python3 freq_data_cleaner.py [json_file] [month] [year]
    """)
else:
    BACKTEST_JSON_PATH = sys.argv[1]
    BACKTEST_MONTH = sys.argv[2]
    BACKTEST_YEAR = sys.argv[3]
    clean_json()