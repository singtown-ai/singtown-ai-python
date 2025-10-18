import time
import os
import argparse
import tempfile
import zipfile
from .client import SingTownAIClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_interval", type=float, default=3.0)
    parser.add_argument("--mock", type=bool, default=False)
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
            mock=args.mock,
        ) as client:
            client.download_trained_file("model_path")
            for i in range(args.epochs):
                client.log(f"train epoch: {i}")
                metrics.append({"epoch": i, "accuracy": i * 10})
                client.upload_metrics(metrics)
                time.sleep(args.train_interval)
            client.upload_results_zip(uploadfile_path)
    finally:
        os.close(upload_fd)
        os.remove(uploadfile_path)
