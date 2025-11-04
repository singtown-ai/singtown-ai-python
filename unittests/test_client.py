from singtown_ai import SingTownAIClient, file_watcher
from .conftest import create_image_files

import zipfile
import time
import pytest
import json
import os


def test_server_error():
    from requests.exceptions import HTTPError
    import requests_mock

    with requests_mock.Mocker() as m:
        m.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=500)
        with pytest.raises(HTTPError):
            SingTownAIClient()


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_update_metrics(request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    metrics = []
    for i in range(5):
        metrics.append({"epoch": i, "accuracy": 0.8, "loss": 0.2})
    client = SingTownAIClient(mock_task_url=task_file)
    client.update_metrics(metrics)
    assert len(client.task.metrics) == 5


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_update_metrics_empty(request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    metrics = []
    client = SingTownAIClient(mock_task_url=task_file)
    client.update_metrics(metrics)
    assert len(client.task.metrics) == 0


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_watch_metrics(tmp_path, request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    metricsfile = tmp_path / "metrics.csv"
    with open(metricsfile, "w") as f:
        f.write("epoch,accuracy,loss\n")
        for i in range(5):
            f.write(f"{i},0.8,0.2\n")

    client = SingTownAIClient(mock_task_url=task_file)

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


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_upload_results_zip_strpath(tmp_path, request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    uploadfile = tmp_path / "result.zip"
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(mock_task_url=task_file)
    client.upload_results_zip(str(uploadfile))


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_upload_results_zip_not_exist(tmp_path, request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    uploadfile = tmp_path / "result.zip"
    client = SingTownAIClient(mock_task_url=task_file)
    with pytest.raises(FileNotFoundError):
        client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_upload_results_zip_pathlike(tmp_path, request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    uploadfile = tmp_path / "result.zip"
    with zipfile.ZipFile(uploadfile, "w") as zf:
        zf.writestr("best.tflite", "content")

    client = SingTownAIClient(mock_task_url=task_file)
    client.upload_results_zip(uploadfile)


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_log(request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    client = SingTownAIClient(mock_task_url=task_file)
    assert len(client.task.logs) == 0
    client.log("train started")
    assert len(client.task.logs) == 1
    assert client.task.logs[0].content == "train started\n"


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
@pytest.mark.parametrize("dataset_fixture", ["dataset_cf_file", "dataset_od_file"])
def test_dataset(request, task_fixture, dataset_fixture):
    task_file = request.getfixturevalue(task_fixture)
    dataset_file = request.getfixturevalue(dataset_fixture)
    client = SingTownAIClient(
        mock_task_url=task_file,
        mock_dataset_url=dataset_file,
    )
    assert len(client.dataset) == 20


@pytest.mark.parametrize("task_fixture", ["task_cf_file", "task_od_file"])
def test_dataset_none(request, task_fixture):
    task_file = request.getfixturevalue(task_fixture)
    client = SingTownAIClient(
        mock_task_url=task_file,
        mock_dataset_url=[],
    )
    assert len(client.dataset) == 0


def test_download_image(tmp_path, task_cf_file, dataset_cf_file):
    client = SingTownAIClient(
        mock_task_url=task_cf_file,
        mock_dataset_url=dataset_cf_file,
    )
    annotation = client.dataset[0]
    folder = tmp_path / "images"
    filepath = client.download_image(annotation.url, folder)
    assert filepath.exists()
    assert filepath.read_bytes() == b"fake image content"


def test_download_image_exists(tmp_path, task_cf_file, dataset_cf_file):
    from pathlib import Path

    client = SingTownAIClient(
        mock_task_url=task_cf_file,
        mock_dataset_url=dataset_cf_file,
    )
    annotation = client.dataset[0]
    folder = tmp_path / "images"
    dst = folder / Path(annotation.url).name
    with open(dst, "wb") as f:
        f.write(b"fake image content not exists")
    assert dst.exists()

    filepath = client.download_image(annotation.url, folder)
    assert dst.absolute() == filepath.absolute()
    assert filepath.read_bytes() == b"fake image content not exists"


def test_export_class_folder_default(tmp_path, task_cf_file, dataset_cf_file):
    client = SingTownAIClient(
        mock_task_url=task_cf_file,
        mock_dataset_url=dataset_cf_file,
    )
    export_path = tmp_path / "dataset"
    client.export_class_folder(export_path)
    assert len(os.listdir(export_path / "TRAIN/cat")) == 7
    assert len(os.listdir(export_path / "TRAIN/dog")) == 7
    assert len(os.listdir(export_path / "VALID/cat")) == 2
    assert len(os.listdir(export_path / "VALID/dog")) == 2
    assert len(os.listdir(export_path / "TEST/cat")) == 1
    assert len(os.listdir(export_path / "TEST/dog")) == 1


def test_export_class_folder_strpath(tmp_path, task_cf_file, dataset_cf_file):
    client = SingTownAIClient(
        mock_task_url=task_cf_file,
        mock_dataset_url=dataset_cf_file,
    )
    export_path = tmp_path / "dataset"
    client.export_class_folder(str(export_path))
    assert len(os.listdir(export_path / "TRAIN/cat")) == 7
    assert len(os.listdir(export_path / "TRAIN/dog")) == 7
    assert len(os.listdir(export_path / "VALID/cat")) == 2
    assert len(os.listdir(export_path / "VALID/dog")) == 2
    assert len(os.listdir(export_path / "TEST/cat")) == 1
    assert len(os.listdir(export_path / "TEST/dog")) == 1


def test_export_class_folder_typeerror(tmp_path, task_od_file):
    client = SingTownAIClient(mock_task_url=task_od_file)
    export_path = tmp_path / "dataset"
    with pytest.raises(RuntimeError):
        client.export_class_folder(export_path)


def test_export_yolo_folder(tmp_path, task_od_file, dataset_od_file):
    client = SingTownAIClient(
        mock_task_url=task_od_file,
        mock_dataset_url=dataset_od_file,
    )
    export_path = tmp_path / "dataset"
    client.export_yolo(export_path)
    assert len(os.listdir((export_path / "images/TRAIN"))) == 14
    assert len(os.listdir((export_path / "images/VALID"))) == 4
    assert len(os.listdir((export_path / "images/TEST"))) == 2
    assert len(os.listdir((export_path / "labels/TRAIN"))) == 14
    assert len(os.listdir((export_path / "labels/VALID"))) == 4
    assert len(os.listdir((export_path / "labels/TEST"))) == 2


def test_export_yolo_yaml(tmp_path, task_od_file, dataset_od_file):
    import yaml
    from pathlib import Path

    client = SingTownAIClient(
        mock_task_url=task_od_file,
        mock_dataset_url=dataset_od_file,
    )
    export_path = tmp_path / "dataset"
    client.export_yolo(export_path)
    with open(export_path / "data.yaml") as f:
        data = yaml.safe_load(f)
    assert data["path"] == str(Path(export_path).absolute())
    assert data["train"] == "images/TRAIN"
    assert data["val"] == "images/VALID"
    assert data["test"] == "images/TEST"
    assert data["nc"] == len(client.task.project.labels)
    assert data["names"] == client.task.project.labels


def test_export_yolo_typeerror(tmp_path, task_cf_file, dataset_od_file):
    client = SingTownAIClient(
        mock_task_url=task_cf_file,
        mock_dataset_url=dataset_od_file,
    )
    export_path = tmp_path / "dataset"
    with pytest.raises(RuntimeError):
        client.export_yolo(export_path)


def test_export_yolo_multibox(tmp_path, task_od_file):
    dataset = [
        {
            "url": f"{tmp_path}/images/cat.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {"label": "cat", "xmin": 0.2, "ymin": 0.01, "xmax": 0.3, "ymax": 0.4},
                {"label": "dog", "xmin": 0.6, "ymin": 0.03, "xmax": 0.8, "ymax": 0.9},
            ],
        },
    ]
    p = tmp_path / "MOCK_DATASET_OD.json"
    p.write_text(json.dumps(dataset))
    create_image_files(dataset)

    client = SingTownAIClient(
        mock_task_url=task_od_file,
        mock_dataset_url=p,
    )
    export_path = tmp_path / "dataset"
    client.export_yolo(export_path)
    with open(export_path / "labels/TRAIN" / "cat.0.txt", "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert lines[0].strip() == "0 0.250000 0.205000 0.100000 0.390000"
        assert lines[1].strip() == "1 0.700000 0.465000 0.200000 0.870000"


def test_export_yolo_label_error(tmp_path, task_od_file):
    dataset = [
        {
            "url": f"{tmp_path}/images/cat.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {"label": "c", "xmin": 0.2, "ymin": 0.01, "xmax": 0.3, "ymax": 0.4},
            ],
        },
    ]
    p = tmp_path / "MOCK_DATASET_OD.json"
    p.write_text(json.dumps(dataset))
    create_image_files(dataset)

    client = SingTownAIClient(
        mock_task_url=task_od_file,
        mock_dataset_url=p,
    )
    export_path = tmp_path / "dataset"
    with pytest.raises(ValueError):
        client.export_yolo(export_path)


def test_export_yolo_width_height_negative(tmp_path, task_od_file):
    dataset = [
        {
            "url": f"{tmp_path}/images/cat.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {"label": "cat", "xmin": 0.2, "ymin": 0.01, "xmax": 0.1, "ymax": 0.4},
            ],
        },
    ]
    p = tmp_path / "MOCK_DATASET_OD.json"
    p.write_text(json.dumps(dataset))
    create_image_files(dataset)

    client = SingTownAIClient(
        mock_task_url=task_od_file,
        mock_dataset_url=p,
    )
    export_path = tmp_path / "dataset"
    with pytest.raises(ValueError):
        client.export_yolo(export_path)
