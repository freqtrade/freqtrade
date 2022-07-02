import subprocess
import watchdog.events
import watchdog.observers

from wao.brain_config import BrainConfig


class _Error_Watcher(watchdog.events.PatternMatchingEventHandler):

    def __init__(self):
        watchdog.events.PatternMatchingEventHandler.__init__(self,
                                                             ignore_directories=False, case_sensitive=False)

    def on_created(self, event):
        print("_Error_Watcher:on_created: file name = " + str(event.src_path))
        file = str(event.src_path)

        error_check_command = "grep -i error " + file + " | grep -i exception " + file
        result = subprocess.Popen([error_check_command],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
        out, err = result.communicate()
        out_put_string = out.decode('latin-1')
        if out_put_string != "":
            stop_bot_command = "python3 " + BrainConfig.EXECUTION_PATH + "/stop_bot.py " + str(
                BrainConfig.MODE) + " " + out_put_string.split("\n")[0].replace("_", "")\
                .replace(": ", ":").replace(" ", "#").replace("(", "").replace(")", "")
            print(stop_bot_command)
            result_log = subprocess.Popen([stop_bot_command],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

            out, err = result_log.communicate()
            out_put = out.decode('latin-1')
            print(out_put)

    def on_modified(self, event):
        print("_Error_Watcher:on_modified: file name = " + str(event.src_path))
        file = str(event.src_path)

        error_check_command = "grep -i error " + file + " | grep -i exception " + file
        result = subprocess.Popen([error_check_command],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
        out, err = result.communicate()
        out_put_string = out.decode('latin-1')
        if out_put_string != "":
            stop_bot_command = "python3 " + BrainConfig.EXECUTION_PATH + "/stop_bot.py " + str(
                BrainConfig.MODE) + " " + out_put_string.split("\n")[0].replace("_", "").\
                replace(": ", ":").replace(" ", "#").replace("(", "").replace(")", "")
            result_log = subprocess.Popen([stop_bot_command],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

            out, err = result_log.communicate()
            out_put = out.decode('latin-1')
            print(out_put)
