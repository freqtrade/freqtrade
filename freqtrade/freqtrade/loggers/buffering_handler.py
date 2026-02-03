from logging.handlers import BufferingHandler


class FTBufferingHandler(BufferingHandler):
    def flush(self):
        """
        Override Flush behaviour - we keep half of the configured capacity
        otherwise, we have moments with "empty" logs.
        """
        self.acquire()
        try:
            # Keep half of the records in buffer.
            records_to_keep = -int(self.capacity / 2)
            self.buffer = self.buffer[records_to_keep:]
        finally:
            self.release()
