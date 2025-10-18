# SingTown AI Python SDK

This SDK is designed to interact with **SingTown AI Cloud Service** or **SingTown AI Standalone**(self-hosted).

## Installation

```bash
pip install singtown_ai
```

## Usage
### Login Configuration

* **SingTown AI Cloud Service**: The `host` is `"https://ai.singtown.com"`.
* **SingTown AI Standalone** (self-hosted): The `host` is something like `"http://127.0.0.1:8000"`.

You can obtain the `token` and `task_id` from **Project -> Information**.

#### Environment Variables:

```bash
export SINGTOWN_AI_HOST="https://ai.singtown.com"  # Or the cloud service URL
export SINGTOWN_AI_TOKEN="your token"            # Your token
export SINGTOWN_AI_TASK_ID="your id"             # Your task ID
```

#### Alternatively, set them directly in code:

```python
SingTownAiClient(
  host="https://ai.singtown.com",  # Or the cloud service URL
  token="your token",            # Your token
  task_id="your id"              # Your task ID
)
```

### Dry Run

```bash
python -m singtown_ai.dryrun --host=http://127.0.0.1:8000 --token=012345 --task_id=1
```

* This command will simulate 10s train task.

### Basic Usage

```python
from singtown_ai import SingTownAiClient

with SingTownAiClient() as client:
    pass  # Insert your code here
```

* This will periodically update the running status. After finished, it will post a "training succeeded" status. If an error occurs, it will post a "training failed" status.

### Mock Usage

```python
from singtown_ai import SingTownAiClient

with SingTownAiClient(mock=True) as client:
    pass  # Insert your code here
```

* Set mock=True, Will mock demo task and dataset, this is useful for debugging.

### Uploading Metrics

```python
metrics = [
    {"epoch": 0, "accuracy": 0.8, "loss": 0.2},
    {"epoch": 1, "accuracy": 0.9, "loss": 0.1},
]
with SingTownAiClient() as client:
    client.upload_metrics(metrics)
```

* The field names in `metrics` are not restricted, and they will appear on the Metrics page in SingTown AI.

### Watching `metrics.csv`

```python
with SingTownAiClient(metrics_file="metrics.csv") as client:
    pass  # Insert your code here
```

* Every 3 seconds, the SDK will parse the `metrics.csv` and upload data.

### Posting Logs

```python
with SingTownAiClient() as client:
    import time
    for i in range(100):
        client.log(f"epoch: {i}")
        time.sleep(0.1)
```

* This will upload log strings, posting them every 3 seconds.

### Download Trained Files from Server

```python
with SingTownAiClient() as client:
    client.download_trained_file("folder")
```

* This method will download the trained file and automatically extract it into the specified folder.

### Uploading Result Files

```python
with SingTownAiClient() as client:
    client.upload_results_zip("your.zip")
```

* This method uploads a `.zip` result file.

### Run Subprocess Command

```python
with SingTownAiClient() as client:
    client.run_subprocess("echo hello world!")
    client.run_subprocess("python3 train.py", ignore_stdout=True)
```

* This method will run subprocess and log stdout and stderr.
* If `ignore_stdout=True` , will not log stdout.