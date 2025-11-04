import os
import threading
import time
import yaml
import requests_mock
from pathlib import Path
from typing import List
from os import PathLike
from .type import Annotation, LogEntry, TaskResponse
import json
import fsspec


class SingTownAIClient:
    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        task_id: str | None = None,
        mock_task_url: str | PathLike | None = None,
        mock_dataset_url: str | PathLike | None = None,
    ):
        self.host = host or os.getenv("SINGTOWN_AI_HOST", "https://ai.singtown.com")
        self.token = token or os.getenv("SINGTOWN_AI_TOKEN", "0123456")
        self.task_id = task_id or os.getenv("SINGTOWN_AI_TASK_ID", "0")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self._request_lock = threading.RLock()

        mock_task_url = mock_task_url or os.getenv("SINGTOWN_AI_MOCK_TASK_URL")
        mock_dataset_url = mock_dataset_url or os.getenv("SINGTOWN_AI_MOCK_DATASET_URL")

        self.mocker = requests_mock.Mocker(real_http=not mock_task_url)
        if mock_task_url:
            self.__setup_mock(mock_task_url, mock_dataset_url)
        self.task = self.__get_task()
        self.dataset = self.__get_dataset()

    def __setup_mock(self, mock_task_url, mock_dataset_url):
        with fsspec.open(str(mock_task_url), "r") as f:
            mock_task_data = TaskResponse(**json.load(f))
        try:
            with fsspec.open(str(mock_dataset_url), "r") as f:
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

    def export_class_folder(self, dataset_path: str | PathLike):
        if self.task.project.type != "CLASSIFICATION":
            raise RuntimeError("export_class_folder only support CLASSIFICATION task")
        for annotation in self.dataset:
            folder = Path(dataset_path) / annotation.subset / annotation.classification
            self.download_image(annotation.url, folder)

    def export_yolo(self, dataset_path: str | PathLike):
        if self.task.project.type != "OBJECT_DETECTION":
            raise RuntimeError("export_yolo only support OBJECT_DETECTION task")

        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        with open(dataset_path / "data.yaml", "w", encoding="utf-8") as f:
            datayaml = {
                "path": str(dataset_path.absolute()),
                "train": "images/TRAIN",
                "val": "images/VALID",
                "test": "images/TEST",
                "nc": len(self.task.project.labels),
                "names": self.task.project.labels,
            }
            yaml.dump(datayaml, f, allow_unicode=True, sort_keys=False)

        for annotation in self.dataset:
            images_subset_path = dataset_path / "images" / annotation.subset
            images_subset_path.mkdir(parents=True, exist_ok=True)
            image_path = self.download_image(annotation.url, images_subset_path)

            labels_subset_path = dataset_path / "labels" / annotation.subset
            labels_subset_path.mkdir(parents=True, exist_ok=True)

            label_filename = labels_subset_path / (image_path.stem + ".txt")
            with open(label_filename, "w") as f:
                for box in annotation.object_detection:
                    cx = (box.xmin + box.xmax) / 2
                    cy = (box.ymin + box.ymax) / 2
                    w = box.xmax - box.xmin
                    h = box.ymax - box.ymin
                    if not (
                        (0 <= cx <= 1)
                        and (0 <= cy <= 1)
                        and (0 <= w <= 1)
                        and (0 <= h <= 1)
                    ):
                        raise ValueError(
                            f"(cx, cy, w, h) must be between 0 and 1! cx: {cx}, cy: {cy}, w: {w}, h: {h}"
                        )
                    class_id = self.task.project.labels.index(box.label)
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
