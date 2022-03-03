# import json
# import datetime
# import time
# import os
# import sys
# from config import Config

# def clean_json():
#     print("clean_json: json_path = " + Config.BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)
#     file = open(Config.BACKTEST_DOWNLOADED_JSON_DATA_FILE_PATH)
#     list = []
#     data = json.load(file)
#     for datas in data:
#         unix = datas[0]/1000
#         unix = int(unix)
#         date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(unix)))
#         date = datetime.datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")
#         year = date.year
#         month = date.month
#         if year == int(Config.BACKTEST_DATA_CLEANER_YEAR) and month == int(Config.BACKTEST_DATA_CLEANER_MONTH_INDEX) + 1:
#             list.append(datas)
#     json_object = json.dumps(list)
#     file.close()
#     write_to_json(json_object)
