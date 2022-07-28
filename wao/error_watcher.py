import subprocess
import watchdog.events
import watchdog.observers
import sys

from wao.brain_config import BrainConfig


class Error_Watcher(watchdog.events.PatternMatchingEventHandler):

    def __init__(self):
        watchdog.events.PatternMatchingEventHandler.__init__(self,
                                                             ignore_directories=False, case_sensitive=False)

    def do_tail_cmd(self, file_name):
        error_check_command = "tail " + file_name
        result = subprocess.Popen([error_check_command],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
        out, err = result.communicate()
        out_put_string = out.decode('latin-1')
        print("out_put_string: " + out_put_string)
        list_of_lines = self.string_to_list(out_put_string)
        return list_of_lines

    def string_to_list(self, string):
        return string.split("\n")

    def get_error_line(self, list_of_lines):
        if len(list_of_lines) > 0:
            for line in list_of_lines:
                if "error" in str(line).lower() or "exception" in str(line).lower():
                    return str(line)
        return None

    def on_created(self, event):
        file_name = str(event.src_path)

        error_check_command = "grep error " + file_name + " && grep exception " \
                              + file_name + " && grep Error " + file_name + " && grep Exception " + file_name \
                              + file_name + " && grep ERROR " + file_name + " && grep EXCEPTION " + file_name
        result = subprocess.Popen([error_check_command],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
        out, err = result.communicate()
        out_put_string = out.decode('latin-1')
        if not self.__freqtrade_error_case(out_put_string):
            stop_bot_command = "python3 " + BrainConfig.EXECUTION_PATH + "/stop_bot.py " + str(
                BrainConfig.MODE) + " " + out_put_string.split("\n")[0].replace("_", "") \
                                   .replace(": ", ":").replace(" ", "#").replace("(", "").replace(")", "")
            result_log = subprocess.Popen([stop_bot_command],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

            out, err = result_log.communicate()
            out_put = out.decode('latin-1')

    def on_modified(self, event):
        file_name = str(event.src_path)

        list_of_lines = self.do_tail_cmd(file_name)
        print("list_of_lines: " + str(list_of_lines))
        error_line = self.get_error_line(list_of_lines)
        if not self.__freqtrade_error_case(error_line):
            stop_bot_command = "python3 " + BrainConfig.EXECUTION_PATH + "/stop_bot.py " + str(
                BrainConfig.MODE) + " " + error_line.split("\n")[0].replace("_", "") \
                                   .replace(": ", ":").replace(" ", "#").replace("(", "").replace(")", "")
            result_log = subprocess.Popen([stop_bot_command],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

            out, err = result_log.communicate()
            out_put = out.decode('latin-1')

    def __freqtrade_error_case(self, out_put_string):
        lower_string = out_put_string.lower()
        return "freqtrade" in lower_string and ("warning" in lower_string or "error" in lower_string)
