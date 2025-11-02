from singtown_ai import file_watcher
import time


def test_watcher_path(tmp_path):
    history = []
    file = tmp_path / "test.txt"
    with open(file, "w") as f:
        f.write("test1")

    @file_watcher(file, interval=0.1)
    def file_on_change(content: str):
        history.append(content)

    time.sleep(0.2)
    assert history == ["test1"]


def test_watcher_str(tmp_path):
    history = []
    file = tmp_path / "test.txt"
    with open(file, "w") as f:
        f.write("test1")

    @file_watcher(str(file), interval=0.1)
    def file_on_change(content: str):
        history.append(content)

    time.sleep(0.2)
    assert history == ["test1"]


def test_watcher_empty(tmp_path):
    history = []
    file = tmp_path / "test.txt"
    with open(file, "w") as f:
        f.write("")

    @file_watcher(file, interval=0.1)
    def file_on_change(content: str):
        history.append(content)

    time.sleep(0.2)
    assert history == [""]


def test_watcher_not_exists():
    history = []

    @file_watcher("not_exists.txt", interval=0.1)
    def file_on_change(content: str):
        history.append(content)

    time.sleep(0.2)
    assert history == []


def test_watcher_twice(tmp_path):
    history = []
    file = tmp_path / "test.txt"
    with open(file, "w") as f:
        f.write("test1")

    @file_watcher(file, interval=0.1)
    def file_on_change(content: str):
        history.append(content)

    time.sleep(0.2)
    with open(file, "w") as f:
        f.write("test2")

    time.sleep(0.2)
    assert history == ["test1", "test2"]


def test_watcher_same(tmp_path):
    history = []
    file = tmp_path / "test.txt"
    with open(file, "w") as f:
        f.write("test1")

    @file_watcher(file, interval=0.1)
    def file_on_change(content: str):
        history.append(content)

    time.sleep(0.2)
    with open(file, "w") as f:
        f.write("test1")

    time.sleep(0.2)
    assert history == ["test1"]


def test_watcher_folder(tmp_path):
    history = []

    @file_watcher(tmp_path, interval=0.1)
    def file_on_change(content: str):
        history.append(content)

    time.sleep(0.2)
    assert history == []
