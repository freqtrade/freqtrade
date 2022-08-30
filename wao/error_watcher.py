import subprocess
import threading

import watchdog.events
import watchdog.observers
import sys

from wao.brain_config import BrainConfig

sys.path.append(BrainConfig.EXECUTION_PATH)
from config import Config


def is_freqtrade_error(error_line):
    lower_string = error_line.lower()
    return "freqtrade" in lower_string and ("warning" in lower_string or "error" in lower_string)


def stop_bot(error_line):
    stop_bot_command = "python3 " + BrainConfig.EXECUTION_PATH + "/stop_bot.py " + str(
        BrainConfig.MODE) + " " + Config.BRAIN + " " + error_line.split("\n")[0].replace("_", "") \
                           .replace(": ", ":").replace(" ", "#").replace("(", "").replace(")", "")
    result_log = subprocess.Popen([stop_bot_command],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

    out, err = result_log.communicate()
    out_put = out.decode('latin-1')


def string_to_list(string):
    return list(string.split("\n"))


def get_tail_cmd_result(file_name):
    tail_command = "tail " + file_name
    result = subprocess.Popen([tail_command],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    out, err = result.communicate()
    out_put_string = out.decode('latin-1')
    return string_to_list(out_put_string)


def check_condition(file_name):
    threading.Thread(target=__check_condition, args=(file_name,)).start()


def __check_condition(file_name):
    error_line = get_error_line(file_name)
    if error_line is not None and not is_freqtrade_error(error_line):
        stop_bot(error_line)


def get_error_line(file_name):
    list_of_lines = get_tail_cmd_result(file_name)
    if len(list_of_lines) > 0:
        for line in list_of_lines:
            line_str = str(line)
            line_lower = line_str.lower()
            if "error" in line_lower or "exception" in line_lower:
                return line_str
    return None


class Error_Watcher(watchdog.events.PatternMatchingEventHandler):

    def __init__(self):
        watchdog.events.PatternMatchingEventHandler.__init__(self,
                                                             ignore_directories=False, case_sensitive=False)

    def on_created(self, event):
        file_name = str(event.src_path)
        check_condition(file_name)

    def on_modified(self, event):
        file_name = str(event.src_path)
        check_condition(file_name)
