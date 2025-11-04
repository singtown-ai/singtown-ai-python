# SingTown AI Python SDK

这个 SDK 旨在与 **SingTown AI 云服务** 或 **SingTown AI 独立部署**（自托管）进行交互。

## 安装

```bash
pip install singtown_ai
```

## 使用说明

### 登录配置

- **SingTown AI 云服务**: `host` 为 `"https://ai.singtown.com"`。
- **SingTown AI 独立部署**（自托管）: `host` 通常是类似 `"http://127.0.0.1:8000"` 的地址。

你可以从 **项目 -> 信息** 获取 `token` 和 `task_id`。

#### 环境变量设置：

```bash
export SINGTOWN_AI_HOST="https://ai.singtown.com"  # 或者云服务的 URL
export SINGTOWN_AI_TOKEN="你的 token"            # 你的 token
export SINGTOWN_AI_TASK_ID="你的 id"             # 你的任务 ID
```

#### 或者直接在代码中设置：

```python
from singtown_ai import SingTownAiClient
client = SingTownAiClient(
  host="https://ai.singtown.com",  # 或者云服务的 URL
  token="你的 token",            # 你的 token
  task_id="你的 id"              # 你的任务 ID
)
```

### 基本用法

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

### 更新指标

```python
metrics = [
    {"epoch": 0, "accuracy": 0.8, "loss": 0.2},
    {"epoch": 1, "accuracy": 0.9, "loss": 0.1},
]
client = SingTownAiClient()
client.update_metrics(metrics)
```

- `metrics.csv` 中的字段名称没有限制，字段会显示在 SingTown AI 的 Metrics 页面中。

### 监控 `metrics.csv`

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

- SDK 每 1 秒钟解析 `metrics.csv` ，并上传指标。

### 日志

```python
client = SingTownAiClient()
client.log("line")
```

### 记录 sys.stdout 和 sys.stderror

```python
from singtown_ai import SingTownAiClient, stdout_watcher

client = SingTownAiClient()

@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")
```

- SDK 每 1 秒钟上传一次 stdout 和 stderr 到日志。

### 上传结果文件

```python
client = SingTownAiClient()
client.upload_results_zip("your.zip")
```

- 该方法会上传一个 `.zip` 格式的结果文件。

### Mock

- mock_task.json

```json
{
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
}
```

- mock_dataset.json

```json
[
  {
    "url": "https://github.com/singtown-ai/singtown-ai-datasets/raw/main/images/cat.0.jpg",
    "subset": "TRAIN",
    "classification": "cat"
  },
  {
    "url": "https://github.com/singtown-ai/singtown-ai-datasets/raw/main/images/cat.1.jpg",
    "subset": "VALID",
    "classification": "cat"
  },
  {
    "url": "https://github.com/singtown-ai/singtown-ai-datasets/raw/main/images/cat.2.jpg",
    "subset": "TEST",
    "classification": "cat"
  },
  {
    "url": "https://github.com/singtown-ai/singtown-ai-datasets/raw/main/images/dog.0.jpg",
    "subset": "TRAIN",
    "classification": "dog"
  },
  {
    "url": "https://github.com/singtown-ai/singtown-ai-datasets/raw/main/images/dog.1.jpg",
    "subset": "VALID",
    "classification": "dog"
  },
  {
    "url": "https://github.com/singtown-ai/singtown-ai-datasets/raw/main/images/dog.2.jpg",
    "subset": "TEST",
    "classification": "dog"
  }
]
```

#### 环境变量设置：

```bash
export SINGTOWN_AI_MOCK_TASK_URL="mock_task.json"
export SINGTOWN_AI_MOCK_DATASET_URL="mock_dataset.json"
```

#### 或者直接在代码中设置：

```python
client = SingTownAiClient(
    mock_task_url="mock_task.json",
    mock_dataset_url="mock_dataset.json",
)
```

- 设置 mock_data, 会使用假的任务和数据集，这对于调试很有用。
