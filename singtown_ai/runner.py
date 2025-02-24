from pydantic import BaseModel
import os
from uuid import UUID
from typing import List
import requests
from enum import Enum
import csv
import time
import subprocess
import tempfile
import zipfile
import argparse


class Annotation(BaseModel):
    name: str
    label: str
    test: bool
    url: str


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    succees = "success"
    failed = "failed"


class Metric(BaseModel):
    epoch: int
    metric: str


class Task(BaseModel):
    id: UUID
    status: TaskStatus
    crated_at: str
    cwd: str
    metrics_path: str | None
    resource_extract: str
    output_file: str
    cmd: List[str]


def read_csv(csv_path: str):
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
    return []


class Runner:
    def __init__(self, host: str, task_id: UUID, token: str):
        self.HOST = host
        self.HEADERS = {"Authorization": f"Bearer {token}"}

        response = self.__request("GET", f"/tasks/{task_id}")
        self.task = Task(**response.json())
        self.__update_status(TaskStatus.running)

    def __request(self, method: str, url: str, json: dict | None = None):
        if method == "GET":
            response = requests.get(
                f"{self.HOST}{url}",
                json=json,
                headers=self.HEADERS,
            )
        elif method == "POST":
            response = requests.post(
                f"{self.HOST}{url}",
                json=json,
                headers=self.HEADERS,
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
                f"{self.HOST}{url}",
                files={"file": f},
                headers=self.HEADERS,
            )
        if not response.ok:
            raise RuntimeError(
                f"Upload {file_path} to {response.url} Error: {response.status_code}"
            )
        else:
            return response

    def __update_status(self, status: TaskStatus):
        self.__request(
            "POST",
            f"/tasks/{self.task.id}",
            {"status": status},
        )

    def succees(self):
        self.__update_status(TaskStatus.succees)

    def failed(self):
        self.__update_status(TaskStatus.failed)

    def metrics(self, metrics: dict):
        self.__request("POST", f"/tasks/{self.task.id}", {"metrics": metrics})

    def log(self, log: str):
        self.__request("POST", f"/tasks/{self.task.id}", {"log": log})
        print("log:", log, end="", flush=True)

    def save(self, result_file: str):
        self.__upload(f"/tasks/{self.task.id}/result", result_file)

    def download_resource(self, extract: str):
        self.log("downloading resource\n")
        filename = None
        os.makedirs(extract, exist_ok=True)
        response = self.__request("GET", f"/tasks/{self.task.id}/resource")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(response.content)
            filename = f.name
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(extract)
        os.remove(filename)
        self.log("download success\n")

    def watch(
        self, args: List[str], cwd: str, metrics_file: str | None, output_file: str
    ):
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        stdout = ""
        last_time = time.time()
        for line in process.stdout:
            stdout += line
            now = time.time()
            if now - last_time > 3:
                self.log(stdout)
                if metrics_file and metrics_file.endswith(".csv"):
                    metrics = read_csv(os.path.join(cwd, metrics_file))
                    self.metrics(metrics)
                stdout = ""
                last_time = now

        code = process.wait()
        if code == 0:
            self.save(os.path.join(cwd, output_file))
            self.succees()
        else:
            for line in process.stderr:
                print(line, end="", flush=True)
            self.failed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SingTown AI Runner")
    parser.add_argument("--host", type=str, help="host")
    parser.add_argument("--task", type=str, help="task id")
    parser.add_argument("--token", type=str, help="task token")
    args = parser.parse_args()

    run = Runner(args.host, args.task, args.token)
    task = run.task

    run.download_resource(task.resource_extract)
    run.watch(task.cmd, task.cwd, task.metrics_path, task.output_file)
