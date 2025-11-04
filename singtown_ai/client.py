import os
import threading
import time
import requests_mock
from pathlib import Path
from typing import List
from os import PathLike
from .type import Annotation, LogEntry, TaskResponse
import json


class SingTownAIClient:
    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        task_id: str | None = None,
        mock_task_path: str | PathLike | None = None,
        mock_dataset_path: str | PathLike | None = None,
    ):
        self.host = host or os.getenv("SINGTOWN_AI_HOST", "https://ai.singtown.com")
        self.token = token or os.getenv("SINGTOWN_AI_TOKEN", "0123456")
        self.task_id = task_id or os.getenv("SINGTOWN_AI_TASK_ID", "0")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self._request_lock = threading.RLock()

        mock_task_path = mock_task_path or os.getenv("SINGTOWN_AI_MOCK_TASK_PATH")
        mock_dataset_path = mock_dataset_path or os.getenv(
            "SINGTOWN_AI_MOCK_DATASET_PATH"
        )

        self.mocker = requests_mock.Mocker(real_http=not mock_task_path)
        if mock_task_path:
            self.__setup_mock(mock_task_path, mock_dataset_path)
        self.task = self.__get_task()
        self.dataset = self.__get_dataset()

    def __setup_mock(self, mock_task_path, mock_dataset_path):
        with open(str(mock_task_path), "r") as f:
            mock_task_data = TaskResponse(**json.load(f))
        try:
            with open(str(mock_dataset_path), "r") as f:
                mock_dataset_data = [Annotation(**item) for item in json.load(f)]
        except FileNotFoundError:
            mock_dataset_data = []

        self.mocker.get(
            f"{self.host}/api/v1/task/tasks/{self.task_id}",
            json=mock_task_data.model_dump(),
        )
        self.mocker.get(
            f"{self.host}/api/v1/task/tasks/{self.task_id}/dataset",
            json=[annotation.model_dump() for annotation in mock_dataset_data],
        )
        self.mocker.post(f"{self.host}/api/v1/task/tasks/{self.task_id}")
        self.mocker.post(f"{self.host}/api/v1/task/tasks/{self.task_id}/result")
        self.mocker.post(f"{self.host}/api/v1/task/tasks/{self.task_id}/logs")

    def request(self, method, url, **kwargs):
        import requests

        with self._request_lock:
            with self.mocker:
                response = requests.request(method, url, **kwargs, headers=self.headers)
                response.raise_for_status()
                return response

    def get(self, url, params=None, **kwargs):
        return self.request("GET", url, params=params, **kwargs)

    def post(self, url, data=None, json=None, **kwargs):
        return self.request("POST", url, data=data, json=json, **kwargs)

    def __get_task(self) -> TaskResponse:
        response = self.get(f"{self.host}/api/v1/task/tasks/{self.task_id}")
        return TaskResponse(**response.json())

    def __post_task(self, json: dict):
        new_task = self.task.model_dump()
        new_task.update(json)
        self.task = TaskResponse(**new_task)
        self.post(f"{self.host}/api/v1/task/tasks/{self.task_id}", json=json)

    def __get_dataset(self) -> List[Annotation]:
        response = self.get(f"{self.host}/api/v1/task/tasks/{self.task_id}/dataset")
        return [Annotation(**item) for item in response.json()]

    def download_image(self, url: str, folder: str | PathLike) -> bytes:
        import fsspec

        fs, path = fsspec.core.url_to_fs(url)
        filename = os.path.basename(path)
        filepath = Path(folder) / filename
        if filepath.exists():
            return filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with fsspec.open(url, "rb") as f:
            content = f.read()
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath

    def log(self, content: str, end: str = "\n"):
        log = LogEntry(timestamp=time.time(), content=content + end)
        response = self.post(
            f"{self.host}/api/v1/task/tasks/{self.task_id}/logs", json=log.model_dump()
        )
        response.raise_for_status()
        self.task.logs.append(log)

    def update_metrics(self, metrics: List[dict]):
        self.__post_task({"metrics": metrics})

    def upload_results_zip(self, file_path: str | PathLike):
        with open(file_path, "rb") as f:
            self.post(
                f"{self.host}/api/v1/task/tasks/{self.task_id}/result",
                files={"file": f},
            )
