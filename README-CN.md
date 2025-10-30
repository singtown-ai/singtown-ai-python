# SingTown AI Python SDK

这个 SDK 旨在与 **SingTown AI 云服务** 或 **SingTown AI 独立部署**（自托管）进行交互。

## 安装

```bash
pip install singtown_ai
```

## 使用说明

### 登录配置

* **SingTown AI 云服务**: `host` 为 `"https://ai.singtown.com"`。
* **SingTown AI 独立部署**（自托管）: `host` 通常是类似 `"http://127.0.0.1:8000"` 的地址。

你可以从 **项目 -> 信息** 获取 `token` 和 `task_id`。

#### 环境变量设置：

```bash
export SINGTOWN_AI_HOST="https://ai.singtown.com"  # 或者云服务的 URL
export SINGTOWN_AI_TOKEN="你的 token"            # 你的 token
export SINGTOWN_AI_TASK_ID="你的 id"             # 你的任务 ID
```

#### 或者直接在代码中设置：

```python
SingTownAiClient(
  host="https://ai.singtown.com",  # 或者云服务的 URL
  token="你的 token",            # 你的 token
  task_id="你的 id"              # 你的任务 ID
)
```

### 模拟运行

```bash
python -m singtown_ai.dryrun --host=https://ai.singtown.com --token=012345 --task_id=1
```

* 这个命令会模拟一个10s的训练任务

### 基本用法

```python
from singtown_ai import SingTownAiClient

with SingTownAiClient() as client:
    pass  # 在这里插入你的代码
```

* 这个代码会定期更新运行状态。完成后，它会发布 "训练成功" 状态。如果出现错误，它会发布 "训练失败" 状态。

### Mock

```python
from singtown_ai import SingTownAiClient
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
with SingTownAiClient(mock=True) as client:
    pass  # 在这里插入你的代码
```

* 设置 mock_data, 会使用假的任务和数据集，这对于调试很有用。


### 上传指标

```python
with SingTownAiClient() as client:
    client.upload_metrics(metrics)
```

* `metrics.csv` 中的字段名称没有限制，字段会显示在 SingTown AI 的 Metrics 页面中。

### 监控 `metrics.csv`

```python
with SingTownAiClient(metrics_file="metrics.csv") as client:
    pass  # 在这里插入你的代码
```

* SDK 每 3 秒钟解析 `metrics.csv` ，并上传指标。

### 发布日志

```python
with SingTownAiClient() as client:
    import time
    for i in range(100):
        client.log(f"epoch: {i}")
        time.sleep(0.1)
```

* 这段代码会上传日志字符串，每 3 秒钟发布一次。

### 上传结果文件

```python
with SingTownAiClient() as client:
    client.upload_results_zip("your.zip")
```

* 该方法会上传一个 `.zip` 格式的结果文件。

### 运行子进程命令

```python
with SingTownAiClient() as client:
    client.run_subprocess("echo hello world!")
    client.run_subprocess("python3 train.py", ignore_stdout=True)
```

* 该方法会运行子进程，并记录stdout和stderr。
* 如果 `ignore_stdout=True` ，不会记录stdout。
