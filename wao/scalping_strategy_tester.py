import json
import requests
from pathlib import Path
import subprocess
import time
import datetime
import csv
import os

coin = 'LTC'
time_range = '1m'
json_file_name = f'/root/workspace2/freqtrade/user_data/data/binance/{coin}_USDT-{time_range}.json'
json_file_content = Path(json_file_name).read_text()
json_file_content = eval(json_file_content)
total_loop_time = len(json_file_content)
minutes_per_day = 1440 if time_range == '1m' else 288
brain_name = "Scalp"
config_file_name = "config_scalp.json"
freqtrade_directory = "/root/workspace2/freqtrade/"
result_saved_directory = "wao/_scalping_results_directory/"
file_format = ".csv"
under_score = "_"
backtest_command = f"source ./.env/bin/activate; freqtrade backtesting -c {config_file_name} -s {brain_name}"


def get_human_readable_time_from_timestamp(unix_time) -> str:
    print("get_human_readable_time_from_timestamp:... ")
    date = str(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(int(unix_time) / 1000)))
    date = datetime.datetime.strptime(str(date), "%d/%m/%Y %H:%M:%S")
    return str(date)


def get_year_from_timestamp(unix_time) -> str:
    print("get_year_from_timestamp:...")
    date = str(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(int(unix_time) / 1000)))
    date = datetime.datetime.strptime(str(date), "%d/%m/%Y %H:%M:%S")
    return str(date.year)


def get_win_rate_percentage_per_year() -> str:
    out_put_to_be_parsed = run_scalping_strategy_command()
    win_rate_percentage = out_put_to_be_parsed.split("|")[19].split(" ")[11].replace(" ", "")
    return win_rate_percentage


def get_year_range() -> str:
    print("get_year_range:...")
    beginning_year = json_file_content[0]
    beginning_year = beginning_year[0]
    beginning_year = get_year_from_timestamp(beginning_year)
    end_year = json_file_content[-1]
    end_year = end_year[0]
    end_year = get_year_from_timestamp(end_year)
    date_to_be_used = str(beginning_year) if beginning_year == end_year else str(beginning_year) + "-" + str(end_year)
    return date_to_be_used


def write_to_csv(list_of_row, win_rate_percentage_per_year):
    print("write_to_csv:... ")
    column_title = ['coin', 'Brain', 'human_readable_time', 'timeframe', 'win_rate_percentage_per_day',
                    'number_of_trades_per_day', 'average_percentage_per_trade', 'cumulative_percentage_per_day']
    year_range = get_year_range()
    if not os.path.exists(result_saved_directory):
        os.makedirs(result_saved_directory)
    csv_file_name = result_saved_directory + brain_name + under_score + coin + under_score + \
                    time_range + under_score + year_range + file_format + under_score + win_rate_percentage_per_year
    with open(csv_file_name, "w") as outfile:
        write = csv.writer(outfile)
        write.writerow(column_title)
        write.writerows(list_of_row)
    outfile.close()
    upload_to_google_drive(csv_file_name)


def upload_to_google_drive(csv_file_name):
    print("upload_to_google_drive:...")
    headers = {
        "Authorization": "Bearer ya29.A0ARrdaM-x6vlmDbjBPx2SQhPYOoT1ym5ZwVQ-wcLsrjqQAKPFd0B15Ks7dGDNnyPCcPFXI4FU9BfUzb1g-gPpQ2UhjUXPn34kvKc5_pR1_UwCFsqWah1j9QqDTkKHyI1yT-qtz_k_WIcLJ0iTguFgLvZdwaEX"}
    para = {
        "name": csv_file_name,
        "parents": ["1tHOq29-W2Sc4XPwjz2oxj8titeOXjiS_"]
    }
    files = {
        'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
        'file': open(csv_file_name, "rb")
    }
    request = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files
    )
    print(request)


def run_scalping_strategy_command():
    print("run_scalping_strategy_command:... ")
    os.chdir(freqtrade_directory)
    result = subprocess.Popen([backtest_command],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    out, err = result.communicate()
    out_put_string = out.decode('latin-1')
    if len(out_put_string) == 0:
        return run_scalping_strategy_command()
    else:
        return out_put_string


def write_to_json(counter):
    print("write_to_json:... ")
    with open(json_file_name, 'w') as outfile:
        writable_file = json_file_content[counter:counter + minutes_per_day]
        outfile.write(str(writable_file))
    outfile.close()


def parse_scalping_strategy_result() -> list:
    print("parse_scalping_strategy_result:... ")
    list_of_rows = []
    counter = 0
    while counter < total_loop_time:
        list_of_row_items = []
        write_to_json(counter)
        out_put_to_be_parsed = run_scalping_strategy_command()
        number_of_trades_per_day = out_put_to_be_parsed.split("|")[13].replace(" ", "")
        average_percentage_per_trade = out_put_to_be_parsed.split("|")[14].replace(" ", "")
        cumulative_percentage_per_day = out_put_to_be_parsed.split("|")[15].replace(" ", "")
        win_rate_percentage_per_day = out_put_to_be_parsed.split("|")[19].split(" ")[17].replace(" ", "") if \
            out_put_to_be_parsed.split("|")[19].split(" ")[17] != " " else out_put_to_be_parsed.split("|")[19].split(" ")[11].replace(" ", "")
        list_of_row_items.append(coin)
        list_of_row_items.append(brain_name)
        unix_time = json_file_content[counter]
        unix_time = unix_time[0]
        readable_time = get_human_readable_time_from_timestamp(unix_time)
        list_of_row_items.append(str(readable_time))
        list_of_row_items.append(time_range)
        list_of_row_items.append(win_rate_percentage_per_day)
        list_of_row_items.append(number_of_trades_per_day)
        list_of_row_items.append(average_percentage_per_trade)
        list_of_row_items.append(cumulative_percentage_per_day)
        list_of_rows.append(list_of_row_items)
        counter += minutes_per_day
    return list_of_rows


def test_scalping_strategy():
    print("test_scalping_strategy:...")
    win_rate_percentage_per_year = get_win_rate_percentage_per_year()
    list_of_rows = parse_scalping_strategy_result()
    write_to_csv(list_of_rows, win_rate_percentage_per_year)


test_scalping_strategy()
print("successfully executed")
