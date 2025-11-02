# SingTown AI Python SDK

This SDK is designed to interact with **SingTown AI Cloud Service** or **SingTown AI Standalone**(self-hosted).

## Installation

```bash
pip install singtown_ai
```

## Usage

### Login Configuration

- **SingTown AI Cloud Service**: The `host` is `"https://ai.singtown.com"`.
- **SingTown AI Standalone** (self-hosted): The `host` is something like `"http://127.0.0.1:8000"`.

You can obtain the `token` and `task_id` from **Project -> Information**.

#### Environment Variables:

```bash
export SINGTOWN_AI_HOST="https://ai.singtown.com"  # Or the cloud service URL
export SINGTOWN_AI_TOKEN="your token"            # Your token
export SINGTOWN_AI_TASK_ID="your id"             # Your task ID
```

#### Alternatively, set them directly in code:

```python
from singtown_ai import SingTownAiClient
client = SingTownAiClient(
  host="https://ai.singtown.com",  # Or the cloud service URL
  token="your token",            # Your token
  task_id="your id"              # Your task ID
)
```

### Basic Usage

```python
import time
from singtown_ai import SingTownAiClient
client = SingTownAiClient()
print(client.task)
client.export_class_folder(export_path) # or client.export_yolo(export_path)

metrics = []
for i in range(10):
    print("Train:", i)
    metrics.append({"epoch": i, "accuracy": i * 10})
    client.update_metrics(metrics)
    time.sleep(1)

client.upload_results_zip(uploadfile)
```

### Mock Usage

```python
mock_data = {
    "task": {
        "project": {
            "labels": ["cat", "dog"],
            "type": "CLASSIFICATION",
        },
        "device": "openmv-cam-h7-plus",
        "model_name": "mobilenet_v2_0.35_128",
        "freeze_backbone": True,
        "batch_size": 16,
        "epochs": 1,
        "learning_rate": 0.001,
        "early_stopping": 3,
        "export_width": 128,
        "export_height": 128,
    },
    "dataset": [
        {
            "url": "https://ai.singtown.com/media/cat.0.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": "https://ai.singtown.com/media/cat.1.jpg",
            "subset": "VALID",
            "classification": "cat",
        },
        {
            "url": "https://ai.singtown.com/media/cat.2.jpg",
            "subset": "TEST",
            "classification": "cat",
        },
        {
            "url": "https://ai.singtown.com/media/dog.0.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": "https://ai.singtown.com/media/dog.1.jpg",
            "subset": "VALID",
            "classification": "dog",
        },
        {
            "url": "https://ai.singtown.com/media/dog.2.jpg",
            "subset": "TEST",
            "classification": "dog",
        },
    ],
}
client = SingTownAiClient(mock_data=mock_data)
```

- Set mock_data, Will mock demo task and dataset, this is useful for debugging.

### Update Metrics

```python
metrics = [
    {"epoch": 0, "accuracy": 0.8, "loss": 0.2},
    {"epoch": 1, "accuracy": 0.9, "loss": 0.1},
]
client = SingTownAiClient()
client.update_metrics(metrics)
```

- The field names in `metrics` are not restricted, and they will appear on the Metrics page in SingTown AI.

### Watching `metrics.csv`

```python
from singtown_ai import SingTownAIClient, file_watcher

client = SingTownAiClient()

@file_watcher("path/to/metrics.csv", interval=3)
def file_on_change(content: str):
    import csv
    from io import StringIO

    metrics = list(csv.DictReader(StringIO(content)))
    if not metrics:
        return
    client.update_metrics(metrics)
```

- Every 1 seconds, the SDK will parse the `metrics.csv` and upload metrics.

### Logging

```python
client = SingTownAiClient()
client.log("line")
```

### Logging sys.stdout and stderror

```python
from singtown_ai import SingTownAiClient, stdout_watcher

client = SingTownAiClient()

@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")
```

- Every 1 seconds, the SDK will upload messages to logging.

### Uploading Result Files

```python
client = SingTownAiClient()
client.upload_results_zip("your.zip")
```

- This method uploads a `.zip` result file.
