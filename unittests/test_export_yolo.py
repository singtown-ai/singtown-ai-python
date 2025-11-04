from singtown_ai import SingTownAIClient, export_yolo
from .conftest import create_image_files
import os
import pytest
import json
import yaml


def test_export_yolo_folder(tmp_path, task_od_file, dataset_od_file):
    client = SingTownAIClient(
        mock_task_url=task_od_file,
        mock_dataset_url=dataset_od_file,
    )
    export_path = tmp_path / "dataset"
    export_yolo(client, export_path)
    assert len(os.listdir((export_path / "images/TRAIN"))) == 14
    assert len(os.listdir((export_path / "images/VALID"))) == 4
    assert len(os.listdir((export_path / "images/TEST"))) == 2
    assert len(os.listdir((export_path / "labels/TRAIN"))) == 14
    assert len(os.listdir((export_path / "labels/VALID"))) == 4
    assert len(os.listdir((export_path / "labels/TEST"))) == 2


def test_export_yolo_yaml(tmp_path, task_od_file, dataset_od_file):
    from pathlib import Path

    client = SingTownAIClient(
        mock_task_url=task_od_file,
        mock_dataset_url=dataset_od_file,
    )
    export_path = tmp_path / "dataset"
    export_yolo(client, export_path)
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
        export_yolo(client, export_path)


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
    export_yolo(client, export_path)
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
        export_yolo(client, export_path)


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
        export_yolo(client, export_path)
