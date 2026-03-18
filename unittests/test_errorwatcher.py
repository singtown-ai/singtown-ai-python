from singtown_ai import error_watcher
import sys


def test_error_watcher_triggers_on_exception():
    history = []
    watcher = error_watcher()

    @watcher
    def on_error(error_message: str):
        history.append(error_message)

    try:
        raise ValueError("test error")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    assert len(history) == 1
    assert "ValueError" in history[0]
    assert "test error" in history[0]

    watcher.close()


def test_error_watcher_multiple_callbacks():
    history1 = []
    history2 = []
    watcher = error_watcher()

    @watcher
    def on_error1(error_message: str):
        history1.append(error_message)

    @watcher
    def on_error2(error_message: str):
        history2.append(error_message)

    try:
        raise RuntimeError("multi callback test")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    assert len(history1) == 1
    assert len(history2) == 1
    assert "RuntimeError" in history1[0]
    assert "multi callback test" in history2[0]

    watcher.close()


def test_error_watcher_callback_exception_does_not_break():
    history = []
    watcher = error_watcher()

    @watcher
    def bad_callback(error_message: str):
        raise Exception("callback error")

    @watcher
    def good_callback(error_message: str):
        history.append(error_message)

    try:
        raise TypeError("type error test")
    except TypeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    assert len(history) == 1
    assert "TypeError" in history[0]

    watcher.close()


def test_error_watcher_preserves_original_excepthook():
    original_calls = []
    original_hook = sys.excepthook

    def custom_hook(exc_type, exc_value, exc_tb):
        original_calls.append(exc_type)

    sys.excepthook = custom_hook
    watcher = error_watcher()

    @watcher
    def on_error(error_message: str):
        pass

    try:
        raise KeyError("key error test")
    except KeyError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    assert len(original_calls) == 1
    assert original_calls[0] is KeyError

    watcher.close()
    sys.excepthook = original_hook


def test_error_watcher_contains_traceback():
    history = []
    watcher = error_watcher()

    @watcher
    def on_error(error_message: str):
        history.append(error_message)

    try:
        raise ValueError("traceback test")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    assert len(history) == 1
    assert "Traceback" in history[0]
    assert "traceback test" in history[0]

    watcher.close()
