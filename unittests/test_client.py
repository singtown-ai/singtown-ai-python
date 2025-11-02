from singtown_ai import (
    SingTownAIClient,
    file_watcher,
    MOCK_TRAIN_CLASSIFICATION,
    MOCK_TRAIN_OBJECT_DETECTION,
)

import zipfile
import time
import pytest
import requests_mock

MOCK_DATAS = [
    MOCK_TRAIN_CLASSIFICATION,
    MOCK_TRAIN_OBJECT_DETECTION,
]


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_server_error(mock_data):
    from requests.exceptions import HTTPError

    with requests_mock.Mocker() as m:
        m.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=500)
        with pytest.raises(HTTPError):
            SingTownAIClient()


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_update_metrics(mock_data):
    metrics = []
    for i in range(5):
        metrics.append({"epoch": i, "accuracy": 0.8, "loss": 0.2})
    client = SingTownAIClient(mock_data=mock_data)
    client.update_metrics(metrics)
    assert len(client.task.metrics) == 5


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_watch_metrics(tmpdir, mock_data):
    metricsfile = tmpdir.join("metrics.csv")
    with open(str(metricsfile), "w") as f:
        f.write("epoch,accuracy,loss\n")
        for i in range(5):
            f.write(f"{i},0.8,0.2\n")

    client = SingTownAIClient(
        mock_data=mock_data,
    )

    @file_watcher(metricsfile, interval=0.1)
    def file_on_change(content: str):
        import csv
        from io import StringIO

        metrics = list(csv.DictReader(StringIO(content)))
        if not metrics:
            return
        client.update_metrics(metrics)

    time.sleep(0.2)
    assert len(client.task.metrics) == 5


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_get_dataset(tmpdir, mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    dataset = client.get_dataset()
    assert len(dataset) != 0


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_upload_results_zip_strpath(tmpdir, mock_data):
    uploadfile = tmpdir.join("result.zip")
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(mock_data=mock_data)
    client.upload_results_zip(str(uploadfile))


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_upload_results_zip_not_exist(tmpdir, mock_data):
    uploadfile = tmpdir.join("result.zip")
    client = SingTownAIClient(mock_data=mock_data)
    with pytest.raises(FileNotFoundError):
        client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_upload_results_zip_pathlike(tmpdir, mock_data):
    uploadfile = tmpdir.join("result.zip")
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(mock_data=mock_data)
    client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("mock_data", MOCK_DATAS)
def test_log(mock_data):
    client = SingTownAIClient(mock_data=mock_data)
    assert len(client.task.logs) == 0
    client.log("train started")
    assert len(client.task.logs) == 1
    assert client.task.logs[0].content == "train started\n"


def test_export_class_folder_default(tmpdir):
    client = SingTownAIClient(mock_data=MOCK_TRAIN_CLASSIFICATION)
    export_path = tmpdir.join("dataset")
    client.export_class_folder(export_path)
    assert len(export_path.join("TRAIN/cat").listdir()) == 7
    assert len(export_path.join("TRAIN/dog").listdir()) == 7
    assert len(export_path.join("VALID/cat").listdir()) == 2
    assert len(export_path.join("VALID/dog").listdir()) == 2
    assert len(export_path.join("TEST/cat").listdir()) == 1
    assert len(export_path.join("TEST/dog").listdir()) == 1


def test_export_class_folder_strpath(tmpdir):
    client = SingTownAIClient(mock_data=MOCK_TRAIN_CLASSIFICATION)
    export_path = tmpdir.join("dataset")
    client.export_class_folder(str(export_path))
    assert len(export_path.join("TRAIN/cat").listdir()) == 7
    assert len(export_path.join("TRAIN/dog").listdir()) == 7
    assert len(export_path.join("VALID/cat").listdir()) == 2
    assert len(export_path.join("VALID/dog").listdir()) == 2
    assert len(export_path.join("TEST/cat").listdir()) == 1
    assert len(export_path.join("TEST/dog").listdir()) == 1


def test_export_class_folder_object_detection(tmpdir):
    client = SingTownAIClient(mock_data=MOCK_TRAIN_OBJECT_DETECTION)
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

    client = SingTownAIClient(mock_data=mock_data)
    export_path = tmpdir.join("dataset")
    client.export_class_folder(export_path)
    assert len(export_path.join("TRAIN/cat").listdir()) == 1


def test_export_yolo_default(tmpdir):
    client = SingTownAIClient(mock_data=MOCK_TRAIN_OBJECT_DETECTION)
    export_path = tmpdir.join("dataset")
    client.export_yolo(export_path)
    assert len(export_path.join("images/TRAIN").listdir()) == 12
    assert len(export_path.join("images/VALID").listdir()) == 6
    assert len(export_path.join("images/TEST").listdir()) == 2
    assert len(export_path.join("labels/TRAIN").listdir()) == 12
    assert len(export_path.join("labels/VALID").listdir()) == 6
    assert len(export_path.join("labels/TEST").listdir()) == 2


def test_export_yolo_yaml(tmpdir):
    import yaml
    from pathlib import Path

    client = SingTownAIClient(mock_data=MOCK_TRAIN_OBJECT_DETECTION)
    export_path = tmpdir.join("dataset")
    client.export_yolo(export_path)
    with open(export_path.join("data.yaml")) as f:
        data = yaml.safe_load(f)
        assert data["path"] == str(Path(export_path).absolute())
        assert data["train"] == "images/TRAIN"
        assert data["val"] == "images/VALID"
        assert data["test"] == "images/TEST"
        assert data["nc"] == len(client.task.project.labels)
        assert data["names"] == client.task.project.labels


def test_export_yolo_classification(tmpdir):
    client = SingTownAIClient(mock_data=MOCK_TRAIN_CLASSIFICATION)
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
    client = SingTownAIClient(mock_data=mock_data)
    export_path = tmpdir.join("dataset")
    client.export_yolo(export_path)
    with open(export_path.join("labels/TRAIN").join("cat.0.txt")) as f:
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
    client = SingTownAIClient(mock_data=mock_data)
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
    client = SingTownAIClient(mock_data=mock_data)
    export_path = tmpdir.join("dataset")
    with pytest.raises(ValueError):
        client.export_yolo(export_path)
