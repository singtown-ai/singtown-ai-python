import threading
from pathlib import Path
from os import PathLike
from io import StringIO
import sys
import atexit
from typing import Union


class file_watcher:
    def __init__(self, file_path: Union[str, PathLike], interval=1.0):
        self.file_path = Path(file_path)
        self.interval = interval
        self.callbacks = []
        self.thread = None
        self.lock = threading.RLock()
        self.last_content = None
        self.start()
        atexit.register(self.action)

    def action(self):
        with self.lock:
            if not self.callbacks:
                return
            if not self.file_path.exists():
                return
            if not self.file_path.is_file():
                return
            with open(self.file_path, "r", encoding="utf-8") as file:
                content = file.read()
                if not content:
                    return
                if self.last_content == content:
                    return
                self.last_content = content
                for callback in self.callbacks:
                    callback(content)

    def start(self):
        self.action()
        self.thread = threading.Timer(self.interval, self.start)
        self.thread.daemon = True
        self.thread.start()

    def __call__(self, func):
        self.callbacks.append(func)
        return func


class stdout_watcher:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.callbacks = []
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.origin_stdout = sys.stdout
        self.origin_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.thread = None
        self.lock = threading.RLock()
        self.start()
        atexit.register(self.action)

    def action(self):
        with self.lock:
            out = self.stdout.getvalue()
            if out:
                self.origin_stdout.write(out)
                self.stdout.truncate(0)
                self.stdout.seek(0)
                for callback in self.callbacks:
                    callback(out)
            err = self.stderr.getvalue()
            if err:
                self.origin_stderr.write(err)
                self.stderr.truncate(0)
                self.stderr.seek(0)
                for callback in self.callbacks:
                    callback(err)

    def start(self):
        self.action()
        self.thread = threading.Timer(self.interval, self.start)
        self.thread.daemon = True
        self.thread.start()

    def __call__(self, func):
        self.callbacks.append(func)
        return func
