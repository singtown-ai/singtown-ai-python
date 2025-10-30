from singtown_ai import (
    SingTownAIClient,
    MOCK_TRAIN_CLASSIFICATION,
    MOCK_TRAIN_OBJECT_DETECTION,
)
import time
import zipfile
import pytest
import requests_mock

MOCK_DATAS = [
    MOCK_TRAIN_CLASSIFICATION,
    MOCK_TRAIN_OBJECT_DETECTION,
]


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_status(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    assert client.task.status == "PENDING"
    with client:
        assert client.task.status == "RUNNING"
    assert client.task.status == "SUCCESS"


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_server_error(mock_data):
    from requests.exceptions import HTTPError

    with requests_mock.Mocker() as m:
        m.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=500)
        with pytest.raises(HTTPError):
            with SingTownAIClient():
                pass


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_exception(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    with pytest.raises(RuntimeError):
        with client:
            raise RuntimeError("test exception")
    assert client.task.status == "FAILED"


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_upload_metrics(mock_data):
    metrics = []
    for i in range(5):
        metrics.append({"epoch": i, "accuracy": 0.8, "loss": 0.2})
    client = SingTownAIClient(mock_data=mock_data)
    with client:
        client.upload_metrics(metrics)
    assert len(client.task.metrics) == 5


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_watch_metrics_strpath(tmpdir, mock_data):
    metricsfile = tmpdir.join("metrics.csv")
    with open(str(metricsfile), "w") as f:
        f.write("epoch,accuracy,loss\n")
        for i in range(5):
            f.write(f"{i},0.8,0.2\n")

    client = SingTownAIClient(metrics_file=str(metricsfile), mock_data=mock_data)
    with client:
        pass
    assert len(client.task.metrics) == 5


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_watch_metrics_not_exist(tmpdir, mock_data):
    metricsfile = tmpdir.join("metrics.csv")
    client = SingTownAIClient(metrics_file=str(metricsfile), mock_data=mock_data)
    with client:
        pass
    assert len(client.task.metrics) == 0


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_watch_metrics_pathlike(tmpdir, mock_data):
    metricsfile = tmpdir.join("metrics.csv")
    client = SingTownAIClient(metrics_file=metricsfile, mock_data=mock_data)
    with client:
        pass
    assert len(client.task.metrics) == 0


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_get_dataset(tmpdir, mock_data):
    with SingTownAIClient(mock_data=mock_data) as client:
        dataset = client.get_dataset()
        assert len(dataset) != 0


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_upload_results_zip_strpath(tmpdir, mock_data):
    uploadfile = tmpdir.join("result.zip")
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(mock_data=mock_data)
    with client:
        client.upload_results_zip(str(uploadfile))


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_upload_results_zip_not_exist(tmpdir, mock_data):
    uploadfile = tmpdir.join("result.zip")
    client = SingTownAIClient(mock_data=mock_data)
    with client:
        with pytest.raises(FileNotFoundError):
            client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_upload_results_zip_pathlike(tmpdir, mock_data):
    uploadfile = tmpdir.join("result.zip")
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(mock_data=mock_data)
    with client:
        client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_logs(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    assert len(client.logs) == 0
    with client:
        client.log("train started")
    assert len(client.logs) == 1
    assert client.logs[0].content == "train started\n"


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_logs_long_time(mock_data):
    client = SingTownAIClient(mock_data=mock_data, upload_interval=0.3)
    with client:
        for i in range(50):
            client.log(f"train epoch: {i}")
            time.sleep(0.01)
    assert len(client.logs) == 2


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_subprocess_status(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    assert client.task.status == "PENDING"
    with client:
        client.run_subprocess("echo Hello, World!")
    assert client.task.status == "SUCCESS"


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_subprocess_error(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    assert client.task.status == "PENDING"
    with pytest.raises(RuntimeError):
        with client:
            client.run_subprocess("invalid_command")
    assert client.task.status == "FAILED"


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_subprocess_logs(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    with client:
        client.run_subprocess("echo Hello, World!")
    assert len(client.logs) == 1
    assert client.logs[0].content == "Hello, World!\n"


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_subprocess_ignore_stdout(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    with client:
        client.run_subprocess("echo Hello, World!", ignore_stdout=True)
    assert len(client.logs) == 0


def test_export_class_folder_default(tmpdir):
    with SingTownAIClient(mock_data=MOCK_TRAIN_CLASSIFICATION) as client:
        export_path = tmpdir.join("dataset")
        client.export_class_folder(export_path)
        assert len(export_path.join("TRAIN/cat").listdir()) == 7
        assert len(export_path.join("TRAIN/dog").listdir()) == 7
        assert len(export_path.join("VALID/cat").listdir()) == 2
        assert len(export_path.join("VALID/dog").listdir()) == 2
        assert len(export_path.join("TEST/cat").listdir()) == 1
        assert len(export_path.join("TEST/dog").listdir()) == 1


def test_export_class_folder_strpath(tmpdir):
    with SingTownAIClient(mock_data=MOCK_TRAIN_CLASSIFICATION) as client:
        export_path = tmpdir.join("dataset")
        client.export_class_folder(str(export_path))
        assert len(export_path.join("TRAIN/cat").listdir()) == 7
        assert len(export_path.join("TRAIN/dog").listdir()) == 7
        assert len(export_path.join("VALID/cat").listdir()) == 2
        assert len(export_path.join("VALID/dog").listdir()) == 2
        assert len(export_path.join("TEST/cat").listdir()) == 1
        assert len(export_path.join("TEST/dog").listdir()) == 1


def test_export_class_folder_object_detection(tmpdir):
    with SingTownAIClient(mock_data=MOCK_TRAIN_OBJECT_DETECTION) as client:
        export_path = tmpdir.join("dataset")
        with pytest.raises(RuntimeError):
            client.export_class_folder(export_path)


def test_export_class_folder_repeat(tmpdir):
    mock_data = MOCK_TRAIN_CLASSIFICATION
    mock_data["dataset"] = [
        {
            "url": "https://ai.singtown.com/media/cat.0.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": "https://ai.singtown.com/media/cat.0.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
    ]

    with SingTownAIClient(mock_data=mock_data) as client:
        export_path = tmpdir.join("dataset")
        client.export_class_folder(export_path)
        assert len(export_path.join("TRAIN/cat").listdir()) == 1


def test_export_yolo_default(tmpdir):
    with SingTownAIClient(mock_data=MOCK_TRAIN_OBJECT_DETECTION) as client:
        export_path = tmpdir.join("dataset")
        client.export_yolo(export_path)
        assert len(export_path.join("images").listdir()) == 20
        assert len(export_path.join("labels").listdir()) == 20


def test_export_yolo_classification(tmpdir):
    with SingTownAIClient(mock_data=MOCK_TRAIN_CLASSIFICATION) as client:
        export_path = tmpdir.join("dataset")
        with pytest.raises(RuntimeError):
            client.export_yolo(export_path)


def test_export_yolo_multibox(tmpdir):
    mock_data = MOCK_TRAIN_OBJECT_DETECTION
    mock_data["dataset"] = [
        {
            "url": "https://ai.singtown.com/media/cat.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {"label": "cat", "xmin": 0.2, "ymin": 0.01, "xmax": 0.3, "ymax": 0.4},
                {"label": "dog", "xmin": 0.6, "ymin": 0.03, "xmax": 0.8, "ymax": 0.9},
            ],
        },
    ]
    with SingTownAIClient(mock_data=mock_data) as client:
        export_path = tmpdir.join("dataset")
        client.export_yolo(export_path)
        with open(export_path.join("labels").join("cat.0.txt")) as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert lines[0].strip() == "0 0.250000 0.205000 0.100000 0.390000"
            assert lines[1].strip() == "1 0.700000 0.465000 0.200000 0.870000"


def test_export_yolo_label_error(tmpdir):
    mock_data = MOCK_TRAIN_OBJECT_DETECTION
    mock_data["dataset"] = [
        {
            "url": "https://ai.singtown.com/media/cat.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {"label": "c", "xmin": 0.2, "ymin": 0.01, "xmax": 0.3, "ymax": 0.4},
            ],
        },
    ]
    with SingTownAIClient(mock_data=mock_data) as client:
        export_path = tmpdir.join("dataset")
        with pytest.raises(ValueError):
            client.export_yolo(export_path)


def test_export_yolo_width_height_negative(tmpdir):
    mock_data = MOCK_TRAIN_OBJECT_DETECTION
    mock_data["dataset"] = [
        {
            "url": "https://ai.singtown.com/media/cat.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {"label": "cat", "xmin": 0.2, "ymin": 0.01, "xmax": 0.1, "ymax": 0.4},
            ],
        },
    ]
    with SingTownAIClient(mock_data=mock_data) as client:
        export_path = tmpdir.join("dataset")
        with pytest.raises(ValueError):
            client.export_yolo(export_path)
