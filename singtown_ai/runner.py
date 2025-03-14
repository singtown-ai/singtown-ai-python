from pydantic import BaseModel
import os
from uuid import UUID
from typing import List
import requests
from enum import Enum
import csv
import time
import json
import subprocess
import tempfile
import zipfile
import argparse

class TaskStatus(str, Enum):
    pending = "PENDING"
    running = "RUNNING"
    success = "SUCCESS"
    failed = "FAILED"


class TaskResponse(BaseModel):
    id: int
    status: TaskStatus
    created: str
    name: str
    epochs: int
    cmd: List[str]


class Runner:
    def __init__(self, host: str, task_id: UUID, token: str, cwd: str, metrics_file, output_file, dataset_dir):
        self.host = host
        self.headers = {"Authorization": f"Bearer {token}"}
        self.cwd = cwd
        self.metrics_file = metrics_file
        self.output_file = output_file
        self.dataset_dir = dataset_dir

        response = self.__request("GET", f"/api/v1/task/tasks/{task_id}")
        self.task = TaskResponse(**response.json())
        self.running()

    def __request(self, method: str, url: str, json: dict | None = None):
        if method == "GET":
            response = requests.get(
                f"{self.host}{url}",
                json=json,
                headers=self.headers,
            )
        elif method == "POST":
            response = requests.post(
                f"{self.host}{url}",
                json=json,
                headers=self.headers,
            )
        else:
            raise RuntimeError(f"request method not support {method}")
        if not response.ok:
            raise RuntimeError(f"request {response.url} error! {response.status_code}")
        else:
            return response

    def __upload(self, url: str, file_path: str):
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.host}{url}",
                files={"file": f},
                headers=self.headers,
            )
        if not response.ok:
            raise RuntimeError(
                f"Upload {file_path} to {response.url} Error: {response.status_code}"
            )
        else:
            return response

    def __update_task(self, json: dict):
        self.__request(
            "POST",
            f"/api/v1/task/tasks/{self.task.id}",
            json,
        )

    def __read_csv(self, csv_path: str) -> List[str]:
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                return [json.dumps(row) for row in reader]  # 每行转换为 JSON 字符串
        return []

    def success(self):
        self.__update_task({"status": TaskStatus.success})

    def failed(self):
        self.__update_task({"status": TaskStatus.failed})

    def running(self):
        self.__update_task({"status": TaskStatus.running})

    def metrics(self):
        if self.metrics_file and self.metrics_file.endswith(".csv"):
            metrics = self.__read_csv(os.path.join(self.cwd, self.metrics_file))
            self.__update_task({"metrics": metrics})

    def log(self, log: str):
        now = time.time()
        self.__update_task({"log": { "timestamp": now, "content": log }})

    def upload(self, result_file: str):
        self.__upload(f"/api/v1/task/tasks/{self.task.id}/result", result_file)

    def download_resource(self):
        self.log("downloading resource\n")
        filename = None
        os.makedirs(self.dataset_dir, exist_ok=True)
        response = self.__request("GET", f"/api/v1/task/tasks/{self.task.id}/pack")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(response.content)
            filename = f.name
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(self.dataset_dir)
        os.remove(filename)
        self.log("download success\n")

    def watch(
        self, args: List[str]
    ):
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
        )
        stdout = ""
        last_time = time.time()
        for line in process.stdout:
            stdout += line
            now = time.time()
            if now - last_time > 3:
                self.log(stdout)
                self.metrics()
                stdout = ""
                last_time = now

        code = process.wait()
        self.metrics()
        
        if code == 0:
            self.upload(os.path.join(self.cwd, self.output_file))
            self.success()
        else:
            for line in process.stderr:
                print(line, end="", flush=True)
            self.failed()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SingTown AI Runner")

    parser.add_argument("--version", action="version", version="0.0.1")
    parser.add_argument("--host", type=str, help="host", required=True)
    parser.add_argument("--task", type=str, help="task id", required=True)
    parser.add_argument("--token", type=str, help="task token", required=True)
    parser.add_argument("--cwd", type=str, help="current working directory", default="./", required=False)
    parser.add_argument("--metrics_file", type=str, help="metrics file", default="metrics.csv", required=False)
    parser.add_argument("--output_file", type=str, help="output file", default="output.zip", required=False)
    parser.add_argument("--dataset_dir", type=str, help="dataset directory", default="dataset", required=False)
    args = parser.parse_args()

    run = Runner(args.host, args.task, args.token, args.cwd, args.metrics_file, args.output_file, args.dataset_dir)
    task = run.task

    run.download_resource()
    run.watch(task.cmd)
