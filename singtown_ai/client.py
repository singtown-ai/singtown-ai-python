import os
import io
import subprocess
import requests
import threading
import csv
import time
import tempfile
import zipfile
import shutil
import requests_mock
from typing import List
from io import StringIO
from .mock import MOCK_DATASET_MAP, MOCK_TASK_MAP
from .type import Annotation, LogEntry, TaskResponse, TaskStatus


class SingTownAIClient:
    def __init__(
        self,
        metrics_file: str | None = None,
        host: str | None = None,
        token: str | None = None,
        task_id: str | None = None,
        upload_interval: float = 3,
        mock: bool = False,
    ):
        self.host = host or os.getenv("SINGTOWN_AI_HOST", "https://ai.singtown.com")
        self.token = token or os.getenv("SINGTOWN_AI_TOKEN", "")
        self.task_id = task_id or os.getenv("SINGTOWN_AI_TASK_ID", "")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.metrics_file = metrics_file
        self.upload_interval = upload_interval
        self.logIO = StringIO()
        self.thread = None
        self._request_lock = threading.RLock()
        self.logs = []
        self.mocker = requests_mock.Mocker(real_http=not mock)
        if mock:
            self.__setup_mock()

        self.task = self.__get_task(self.task_id)

    def __setup_mock(self):
        for task_id in MOCK_TASK_MAP.keys():
            self.mocker.get(
                f"{self.host}/api/v1/task/tasks/{task_id}",
                json=MOCK_TASK_MAP[task_id].model_dump(),
            )
            self.mocker.post(
                f"{self.host}/api/v1/task/tasks/{task_id}",
                json=MOCK_TASK_MAP[task_id].model_dump(),
            )
            self.mocker.post(
                f"{self.host}/api/v1/task/tasks/{task_id}/result",
                json={"status": "success"},
            )
            self.mocker.post(
                f"{self.host}/api/v1/task/tasks/{task_id}/logs",
                json={"status": "success"},
            )
            if MOCK_TASK_MAP[task_id].trained_file:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as zf:
                    zf.writestr("best.onnx", "mock model content")
                self.mocker.get(
                    MOCK_TASK_MAP[task_id].trained_file,
                    content=buffer.getvalue(),
                )
        for task_id in MOCK_DATASET_MAP.keys():
            self.mocker.get(
                f"{self.host}/api/v1/task/tasks/{task_id}/dataset",
                json=[
                    annotation.model_dump() for annotation in MOCK_DATASET_MAP[task_id]
                ],
            )

    def __request(
        self,
        method: str,
        url: str,
        host: str = None,
        data=None,
        json=None,
        files=None,
    ):
        with self._request_lock:
            if host is None:
                host = self.host
            with self.mocker:
                if method == "GET":
                    response = requests.get(
                        f"{host}{url}",
                        data=data,
                        json=json,
                        files=files,
                        headers=self.headers,
                    )
                elif method == "POST":
                    response = requests.post(
                        f"{self.host}{url}",
                        data=data,
                        json=json,
                        files=files,
                        headers=self.headers,
                    )
                else:
                    raise RuntimeError(f"request method not support {method}")
                if not response.ok:
                    raise RuntimeError(
                        f"request {response.url} error! {response.status_code}"
                    )
                else:
                    return response

    def __get_task(self, task_id: str) -> TaskResponse:
        response = self.__request("GET", f"/api/v1/task/tasks/{task_id}")
        return TaskResponse(**response.json())

    def __post_task(self, json: dict):
        new_task = self.task.model_dump()
        new_task.update(json)
        self.task = TaskResponse(**new_task)
        self.__request("POST", f"/api/v1/task/tasks/{self.task_id}", json)

    def get_dataset(self) -> List[Annotation]:
        response = self.__request("GET", f"/api/v1/task/tasks/{self.task_id}/dataset")
        return [Annotation(**item) for item in response.json()]

    def __post_log(self, content: str):
        log = LogEntry(timestamp=time.time(), content=content)
        self.__request(
            "POST",
            f"/api/v1/task/tasks/{self.task_id}/logs",
            json=log.model_dump(),
        )
        self.logs.append(log)

    def upload_metrics(self, metrics: List[dict]):
        self.__post_task({"metrics": metrics})

    def upload_results_zip(self, file_path: str):
        with open(file_path, "rb") as f:
            self.__request(
                "POST", f"/api/v1/task/tasks/{self.task_id}/result", files={"file": f}
            )

    def download_trained_file(self, model_path: str):
        trained_file = self.task.trained_file
        if not trained_file:
            return
        shutil.rmtree(model_path, ignore_errors=True)
        os.makedirs(model_path)
        response = self.__request("GET", trained_file, host="")
        filename = None
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(response.content)
            filename = f.name
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(model_path)
        os.remove(filename)

    def __loop_once(self):
        if self.metrics_file and self.metrics_file.endswith(".csv"):
            metrics = self.__read_csv(self.metrics_file)
            if metrics:
                self.upload_metrics(metrics)

        content = self.logIO.getvalue()
        self.logIO.truncate(0)
        self.logIO.seek(0)
        if content:
            self.__post_log(content)

    def __read_csv(self, csv_path: str) -> List[dict]:
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                return list(reader)
        return []

    def log(self, message: str, end: str = "\n"):
        self.logIO.write(message + end)

    def __watch_log(self):
        self.__loop_once()
        self.thread = threading.Timer(self.upload_interval, self.__watch_log)
        self.thread.daemon = True
        self.thread.start()

    def run_subprocess(self, cmd: str, ignore_stdout=False):
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
        )
        if not ignore_stdout:
            for line in process.stdout:
                self.log(line, end="")

        if process.wait() != 0:
            for line in process.stderr:
                self.log(line, end="")
            raise RuntimeError(f"subprocess {cmd} failed")

    def __enter__(self):
        self.__post_task({"status": TaskStatus.running})
        self.__watch_log()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.thread:
            self.thread.cancel()
            self.thread = None

        self.__loop_once()
        if exc_type is None:
            self.__post_task({"status": TaskStatus.success})
        else:
            self.log(f"Exception: {exc_value}")
            self.__post_task({"status": TaskStatus.failed})
