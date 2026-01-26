import logging
import threading
import queue
import time
from typing import Optional

class BackgroundLogger:
    def __init__(self, log_file: Optional[str] = None, level=logging.INFO):
        self.log_queue = queue.Queue()
        self.logger = logging.getLogger("BackgroundLogger")
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._process_logs, daemon=True)
        self.thread.start()

    def _process_logs(self):
        while not self._stop_event.is_set():
            try:
                record = self.log_queue.get(timeout=0.5)
                self.logger.log(record[0], record[1])
            except queue.Empty:
                continue

    def log(self, level, message):
        self.log_queue.put((level, message))

    def info(self, message):
        self.log(logging.INFO, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message)

    def stop(self):
        self._stop_event.set()
        self.thread.join()

# Usage example:
# logger = BackgroundLogger("agent.log")
# logger.info("Agent started")
# logger.error("Something went wrong")
# ...
# logger.stop()  # On shutdown
