import argparse
import time
import json
import os
from singtown_ai import runner
import requests
from typing import List


def downlaod_images(annotations: List[runner.Annotation], save_path: str):
    os.makedirs(save_path, exist_ok=True)
    for anno in annotations:
        with open(os.path.join(save_path, anno.name), "wb") as f:
            f.write(requests.get(anno.url).content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SingTown AI Trainer Example")
    parser.add_argument("host", type=str, help="host")
    parser.add_argument("task", type=str, help="task id")
    parser.add_argument("token", type=str, help="task token")
    args = parser.parse_args()

    runner.login(args.host, args.token)

    task = runner.retrieve_task(args.task)

    # Retrieve project
    project = runner.retrieve_project(task.id)
    if not project:
        raise Exception("Project not found")
    runner.update_status(task.id, runner.TaskStatus.running)
    downlaod_images(project.annotations, "datasets")
    print(project)

    for epoch in range(task.epochs):
        time.sleep(1)
        # Upload metrics
        metric = json.dumps({"loss": 0.1, "accuracy": 0.9})
        runner.upload_metric(task.id, epoch, metric)
        # Upload log
        runner.upload_log(task.id, f"Epoch {epoch} completed")

    # Upload result
    runner.upload_result(task.id, os.path.join(os.path.dirname(__file__), "result.zip"))
    runner.update_status(task.id, runner.TaskStatus.succees)
