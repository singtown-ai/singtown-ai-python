import os
import threading
import time
import yaml
import requests_mock
from pathlib import Path
from typing import List
from os import PathLike
from .type import Annotation, LogEntry, TaskResponse, MockData


class SingTownAIClient:
    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        task_id: str | None = None,
        mock_data: dict = {},
    ):
        self.host = host or os.getenv("SINGTOWN_AI_HOST", "https://ai.singtown.com")
        self.token = token or os.getenv("SINGTOWN_AI_TOKEN", "0123456")
        self.task_id = task_id or os.getenv("SINGTOWN_AI_TASK_ID", "0")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self._request_lock = threading.RLock()

        self.mock_data = None
        self.mocker = requests_mock.Mocker(real_http=not mock_data)
        if mock_data:
            self.mock_data = MockData(**mock_data)
            self.__setup_mock()
        self.task = self.__get_task()

    def __setup_mock(self):
        self.mocker.get(
            f"{self.host}/api/v1/task/tasks/{self.task_id}",
            json=self.mock_data.task.model_dump(),
        )
        self.mocker.post(
            f"{self.host}/api/v1/task/tasks/{self.task_id}",
            json=self.mock_data.task.model_dump(),
        )
        self.mocker.post(f"{self.host}/api/v1/task/tasks/{self.task_id}/result")
        self.mocker.post(f"{self.host}/api/v1/task/tasks/{self.task_id}/logs")
        self.mocker.get(
            f"{self.host}/api/v1/task/tasks/{self.task_id}/dataset",
            json=[annotation.model_dump() for annotation in self.mock_data.dataset],
        )
        mock_dataset_dir = Path(__file__).parent.joinpath("dataset")
        for file in mock_dataset_dir.glob("*"):
            self.mocker.get(
                f"https://ai.singtown.com/media/{file.name}",
                content=file.read_bytes(),
            )

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

    def get_dataset(self) -> List[Annotation]:
        response = self.get(f"{self.host}/api/v1/task/tasks/{self.task_id}/dataset")
        return [Annotation(**item) for item in response.json()]

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
        dataset = self.get_dataset()
        for annotation in dataset:
            folder = Path(dataset_path) / annotation.subset / annotation.classification
            folder.mkdir(parents=True, exist_ok=True)
            response = self.get(annotation.url)
            with open(folder / Path(annotation.url).name, "wb") as f:
                f.write(response.content)

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

        dataset = self.get_dataset()

        for annotation in dataset:
            response = self.get(annotation.url)

            images_subset_path = dataset_path / "images" / annotation.subset
            labels_subset_path = dataset_path / "labels" / annotation.subset

            images_subset_path.mkdir(parents=True, exist_ok=True)
            labels_subset_path.mkdir(parents=True, exist_ok=True)

            image_filename = images_subset_path / Path(annotation.url).name
            with open(image_filename, "wb") as f:
                f.write(response.content)

            label_filename = labels_subset_path / (image_filename.stem + ".txt")
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
