import os
import subprocess
import threading
import csv
import time
import yaml
import requests_mock
from pathlib import Path
from typing import List
from io import StringIO
from os import PathLike
from .type import Annotation, LogEntry, TaskResponse, MockData


class SingTownAIClient:
    def __init__(
        self,
        metrics_file: str | PathLike | None = None,
        host: str | None = None,
        token: str | None = None,
        task_id: str | None = None,
        upload_interval: float = 3,
        mock_data: dict = {},
    ):
        self.host = host or os.getenv("SINGTOWN_AI_HOST", "https://ai.singtown.com")
        self.token = token or os.getenv("SINGTOWN_AI_TOKEN", "0123456")
        self.task_id = task_id or os.getenv("SINGTOWN_AI_TASK_ID", "0")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.metrics_file = Path(metrics_file) if metrics_file else None
        self.upload_interval = upload_interval
        self.logIO = StringIO()
        self.thread = None
        self._request_lock = threading.RLock()
        self.logs = []
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
        self.mocker.post(
            f"{self.host}/api/v1/task/tasks/{self.task_id}/result",
            json={"status": "success"},
        )
        self.mocker.post(
            f"{self.host}/api/v1/task/tasks/{self.task_id}/logs",
            json={"status": "success"},
        )
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
        response.raise_for_status()
        return TaskResponse(**response.json())

    def __post_task(self, json: dict):
        new_task = self.task.model_dump()
        new_task.update(json)
        self.task = TaskResponse(**new_task)
        response = self.post(f"{self.host}/api/v1/task/tasks/{self.task_id}", json=json)
        response.raise_for_status()

    def __update_status(self, status: str):
        self.__post_task({"status": status})

    def get_dataset(self) -> List[Annotation]:
        response = self.get(f"{self.host}/api/v1/task/tasks/{self.task_id}/dataset")
        response.raise_for_status()
        return [Annotation(**item) for item in response.json()]

    def __post_log(self, content: str):
        log = LogEntry(timestamp=time.time(), content=content)
        response = self.post(
            f"{self.host}/api/v1/task/tasks/{self.task_id}/logs", json=log.model_dump()
        )
        response.raise_for_status()
        self.logs.append(log)

    def upload_metrics(self, metrics: List[dict]):
        self.__post_task({"metrics": metrics})

    def upload_results_zip(self, file_path: str | PathLike):
        with open(file_path, "rb") as f:
            response = self.post(
                f"{self.host}/api/v1/task/tasks/{self.task_id}/result",
                files={"file": f},
            )
            response.raise_for_status()

    def __loop_once(self):
        if self.metrics_file and self.metrics_file.suffix == ".csv":
            metrics = self.__read_csv(self.metrics_file)
            if metrics:
                self.upload_metrics(metrics)

        content = self.logIO.getvalue()
        self.logIO.truncate(0)
        self.logIO.seek(0)
        if content:
            self.__post_log(content)

    def __read_csv(self, csv_path: Path) -> List[dict]:
        if not csv_path.exists():
            return []
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)

    def log(self, message: str, end: str = "\n"):
        print(message, end=end)
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

    def export_class_folder(self, dataset_path: str | PathLike):
        if self.task.project.type != "CLASSIFICATION":
            raise RuntimeError("export_class_folder only support CLASSIFICATION task")
        dataset = self.get_dataset()
        for annotation in dataset:
            folder = Path(dataset_path) / annotation.subset / annotation.classification
            folder.mkdir(parents=True, exist_ok=True)
            response = self.get(annotation.url)
            response.raise_for_status()
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

    def __enter__(self):
        self.__update_status("RUNNING")
        self.__watch_log()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.thread:
            self.thread.cancel()
            self.thread = None

        self.__loop_once()
        if exc_type is None:
            self.__update_status("SUCCESS")
        else:
            self.log(f"Exception: {exc_value}")
            self.__update_status("FAILED")
