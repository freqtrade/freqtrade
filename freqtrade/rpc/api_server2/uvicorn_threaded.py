import contextlib
import threading
import time

import uvicorn


class UvicornServer(uvicorn.Server):
    """
    Multithreaded server - as found in https://github.com/encode/uvicorn/issues/742
    """
    def install_signal_handlers(self):
        """
        In the parent implementation, this starts the thread, therefore we must patch it away here.
        """
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        while not self.started:
            time.sleep(1e-3)

    def cleanup(self):
        self.should_exit = True
        self.thread.join()
