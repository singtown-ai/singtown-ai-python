import time
import os
import argparse
import tempfile
import zipfile
from .client import SingTownAIClient
from .mock import (
    MOCK_TRAIN_CLASSIFICATION,
    MOCK_TRAIN_OBJECT_DETECTION,
)

MOCK_MAP = {
    "MOCK_TRAIN_CLASSIFICATION": MOCK_TRAIN_CLASSIFICATION,
    "MOCK_TRAIN_OBJECT_DETECTION": MOCK_TRAIN_OBJECT_DETECTION,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_interval", type=float, default=3.0)
    parser.add_argument("--mock_data", type=str, default=None)
    args = parser.parse_args()

    upload_fd, uploadfile_path = tempfile.mkstemp(suffix=".zip")
    metrics = []

    with zipfile.ZipFile(uploadfile_path, "w") as zf:
        zf.writestr("best.tflite", "content")
    try:
        with SingTownAIClient(
            host=args.host,
            token=args.token,
            task_id=args.task_id,
            mock_data=MOCK_MAP.get(args.mock_data),
        ) as client:
            for i in range(args.epochs):
                client.log(f"train epoch: {i}")
                metrics.append({"epoch": i, "accuracy": i * 10})
                client.upload_metrics(metrics)
                time.sleep(args.train_interval)
            client.upload_results_zip(uploadfile_path)
    finally:
        os.close(upload_fd)
        os.remove(uploadfile_path)
