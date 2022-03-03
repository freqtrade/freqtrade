import time
import watchdog.events
import watchdog.observers
import os
from pathlib import Path

from user_data.strategies.notifier import post_request
# from user_data.strategies._429_file_util import delete_429_file


class _429_Watcher(watchdog.events.PatternMatchingEventHandler):

    def __init__(self, is_test_mode):
        self.is_test_mode = is_test_mode
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self,
                                                             ignore_directories=True, case_sensitive=False)

    def on_created(self, event):
        file = str(event.src_path)
        if os.path.isfile(file):
            content = Path(file).read_text()
            time.sleep(1)  # this sleep makes the watcher wait to avoid overlapping
            print("sending message on: "+str(content))
            post_request(str(content), True)

    # def on_modified(self, event):
    #     print("_429_Watcher:on_create: file name = " + str(event.src_path))
