from singtown_ai import stdout_watcher
import time
import sys


def test_stdout_watcher():
    history = []

    @stdout_watcher(interval=0.1)
    def on_stdout_write(content: str):
        history.append(content)

    print("Hello, World!")

    time.sleep(0.2)
    assert history == ["Hello, World!\n"]


def test_stderr_watcher():
    history = []

    @stdout_watcher(interval=0.1)
    def on_stderr_write(content: str):
        history.append(content)

    print("Hello, World!", file=sys.stderr)

    time.sleep(0.2)
    assert history == ["Hello, World!\n"]
