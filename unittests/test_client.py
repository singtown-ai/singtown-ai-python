from singtown_ai import SingTownAIClient
import time
import zipfile
import pytest
import requests_mock

TASK_IDS = ["0", "1", "2"]


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_status(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    assert client.task.status == "PENDING"
    with client:
        assert client.task.status == "RUNNING"
    assert client.task.status == "SUCCESS"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_server_error(task_id):
    with requests_mock.Mocker() as m:
        m.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=500)
        with pytest.raises(RuntimeError):
            with SingTownAIClient(task_id=task_id, mock=False):
                pass


def test_reuest_post():
    with pytest.raises(RuntimeError):
        with SingTownAIClient(task_id="0", mock=True) as client:
            client._SingTownAIClient__request("PUT", "")


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_exception(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    with pytest.raises(RuntimeError):
        with client:
            raise RuntimeError("test exception")
    assert client.task.status == "FAILED"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_upload_metrics(task_id):
    metrics = []
    for i in range(5):
        metrics.append({"epoch": i, "accuracy": 0.8, "loss": 0.2})
    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        client.upload_metrics(metrics)
    assert len(client.task.metrics) == 5


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_watch_metrics(tmpdir, task_id):
    metricsfile = tmpdir.join("metrics.csv")
    with open(str(metricsfile), "w") as f:
        f.write("epoch,accuracy,loss\n")
        for i in range(5):
            f.write(f"{i},0.8,0.2\n")

    client = SingTownAIClient(metrics_file=str(metricsfile), task_id=task_id, mock=True)
    with client:
        pass
    assert len(client.task.metrics) == 5


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_watch_metrics_not_exist(tmpdir, task_id):
    metricsfile = tmpdir.join("metrics.csv")
    client = SingTownAIClient(metrics_file=str(metricsfile), task_id=task_id, mock=True)
    with client:
        pass
    assert len(client.task.metrics) == 0


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_watch_metrics_pathlike(tmpdir, task_id):
    from pathlib import Path

    metricsfile = Path(tmpdir).joinpath("metrics.csv")
    client = SingTownAIClient(metrics_file=metricsfile, task_id=task_id, mock=True)
    with client:
        pass
    assert len(client.task.metrics) == 0


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_get_dataset(tmpdir, task_id):
    with SingTownAIClient(task_id=task_id, mock=True) as client:
        dataset = client.get_dataset()
        assert len(dataset) != 0


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_upload_results_zip(tmpdir, task_id):
    uploadfile = tmpdir.join("result.zip")
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        client.upload_results_zip(str(uploadfile))


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_upload_results_zip_not_exist(tmpdir, task_id):
    uploadfile = tmpdir.join("result.zip")
    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        with pytest.raises(FileNotFoundError):
            client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_upload_results_zip_pathlike(tmpdir, task_id):
    uploadfile = tmpdir.join("result.zip")
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("task_id", ["2"])
def test_download_trained_file_exist(tmpdir, task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        client.download_trained_file(tmpdir)
        assert tmpdir.join("best.onnx").exists()


@pytest.mark.parametrize("task_id", ["0", "1"])
def test_download_trained_file_not_exist(tmpdir, task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        client.download_trained_file(tmpdir)
        assert not tmpdir.join("best.onnx").exists()


@pytest.mark.parametrize("task_id", ["0", "1"])
def test_download_trained_file_pathlike(tmpdir, task_id):
    from pathlib import Path

    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        folder = Path(tmpdir)
        client.download_trained_file(folder)
        assert not folder.joinpath("best.onnx").exists()


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_logs(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    assert len(client.logs) == 0
    with client:
        client.log("train started")
    assert len(client.logs) == 1
    assert client.logs[0].content == "train started\n"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_logs_long_time(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True, upload_interval=0.3)
    with client:
        for i in range(50):
            client.log(f"train epoch: {i}")
            time.sleep(0.01)
    assert len(client.logs) == 2


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_subprocess_status(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    assert client.task.status == "PENDING"
    with client:
        client.run_subprocess("echo Hello, World!")
    assert client.task.status == "SUCCESS"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_subprocess_error(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    assert client.task.status == "PENDING"
    with pytest.raises(RuntimeError):
        with client:
            client.run_subprocess("invalid_command")
    assert client.task.status == "FAILED"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_subprocess_logs(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        client.run_subprocess("echo Hello, World!")
    assert len(client.logs) == 1
    assert client.logs[0].content == "Hello, World!\n"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_subprocess_ignore_stdout(task_id):
    client = SingTownAIClient(task_id=task_id, mock=True)
    with client:
        client.run_subprocess("echo Hello, World!", ignore_stdout=True)
    assert len(client.logs) == 0
