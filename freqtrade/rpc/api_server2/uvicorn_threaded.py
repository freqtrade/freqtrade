import contextlib
import threading
import time

import uvicorn


class UvicornServer(uvicorn.Server):
    """
    Multithreaded server - as found in https://github.com/encode/uvicorn/issues/742
    """
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        # try:
        while not self.started:
            time.sleep(1e-3)
            # yield
        # finally:
        #     self.should_exit = True
        #     thread.join()

    def cleanup(self):
        self.should_exit = True
        self.thread.join()
