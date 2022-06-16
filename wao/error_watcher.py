import subprocess
import watchdog.events
import watchdog.observers

from wao.brain_config import BrainConfig


class error_Watcher(watchdog.events.PatternMatchingEventHandler):

    def __init__(self):
        watchdog.events.PatternMatchingEventHandler.__init__(self,
                                                             ignore_directories=False, case_sensitive=False)

    def on_created(self, event):
        print("error_Watcher:on_created: file name = " + str(event.src_path))
        file = str(event.src_path)
        error_check_command = "tail -n 20 " + file + " | grep -i error | grep -i exception"
        result = subprocess.Popen([error_check_command],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
        out, err = result.communicate()
        out_put_string = out.decode('latin-1')
        if out_put_string != "":
            is_test_mode = False if BrainConfig.MODE == "test" else True
            stop_bot_command = "python3 " + BrainConfig.EXECUTION_PATH + " stop_bot.py " + str(
                is_test_mode) + " " + out_put_string.replace(" ", "_").replace("(", "").replace(")", "")
            result_log = subprocess.Popen([stop_bot_command],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

            out, err = result_log.communicate()
            out_put = out.decode('latin-1')
            print("result= " + str(file) + " " + str(out_put))

    def on_modified(self, event):
        print("error_Watcher:on_modified: file name = " + str(event.src_path))
        file = str(event.src_path)

        error_check_command_1 = "grep -i error " + file
        error_check_command_2 = "grep -i exception " + file
        result = subprocess.Popen([error_check_command_1, error_check_command_2],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
        out, err = result.communicate()
        out_put_string = out.decode('latin-1')
        if out_put_string != "":
            is_test_mode = False if BrainConfig.MODE == "test" else True
            stop_bot_command = "python3 " + BrainConfig.EXECUTION_PATH + " stop_bot.py " + str(
                is_test_mode) + " " + out_put_string.split("\n")[0].replace(" ", "_").replace("(", "").replace(")", "")
            result_log = subprocess.Popen([stop_bot_command],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

            out, err = result_log.communicate()
            out_put = out.decode('latin-1')
            print("result= " + str(file) + " " + str(out_put))
