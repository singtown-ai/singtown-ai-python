import pytest
import json
from pathlib import Path


def create_image_files(dataset):
    for annotation in dataset:
        img_path = Path(annotation["url"])
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"fake image content")


@pytest.fixture
def task_cf_file(tmp_path):
    """
    Create a temporary MOCK_TASK_CF.json file and return its path.
    """
    task = {
        "project": {"labels": ["cat", "dog"], "type": "CLASSIFICATION"},
        "device": "openmv-cam-h7-plus",
        "model_name": "mobilenet_v2_0.35_128",
        "freeze_backbone": True,
        "batch_size": 16,
        "epochs": 1,
        "learning_rate": 0.001,
        "early_stopping": 3,
        "export_width": 128,
        "export_height": 128,
        "metrics": [],
        "logs": [],
    }
    p = tmp_path / "MOCK_TASK_CF.json"
    p.write_text(json.dumps(task))
    return str(p)


@pytest.fixture
def task_od_file(tmp_path):
    """
    Create a temporary MOCK_TASK_OD.json file and return its path.
    """
    task = {
        "project": {"labels": ["cat", "dog"], "type": "OBJECT_DETECTION"},
        "device": "singtown-ai-vision-module",
        "model_name": "yolov5s_640",
        "freeze_backbone": True,
        "batch_size": 16,
        "epochs": 1,
        "learning_rate": 0.001,
        "early_stopping": 3,
        "export_width": 640,
        "export_height": 480,
        "metrics": [],
        "logs": [],
    }
    p = tmp_path / "MOCK_TASK_OD.json"
    p.write_text(json.dumps(task))
    return str(p)


@pytest.fixture
def dataset_cf_file(tmp_path):
    """
    Create a temporary MOCK_DATASET_CF.json file and return its path.
    """
    dataset = [
        {
            "url": f"{tmp_path}/images/cat.0.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.1.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.2.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.3.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.4.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.5.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.6.jpg",
            "subset": "TRAIN",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.7.jpg",
            "subset": "VALID",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.8.jpg",
            "subset": "VALID",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/cat.9.jpg",
            "subset": "TEST",
            "classification": "cat",
        },
        {
            "url": f"{tmp_path}/images/dog.0.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.1.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.2.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.3.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.4.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.5.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.6.jpg",
            "subset": "TRAIN",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.7.jpg",
            "subset": "VALID",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.8.jpg",
            "subset": "VALID",
            "classification": "dog",
        },
        {
            "url": f"{tmp_path}/images/dog.9.jpg",
            "subset": "TEST",
            "classification": "dog",
        },
    ]
    p = tmp_path / "MOCK_DATASET_CF.json"
    p.write_text(json.dumps(dataset))
    create_image_files(dataset)
    return str(p)


@pytest.fixture
def dataset_od_file(tmp_path):
    """
    Create a temporary MOCK_DATASET_OD.json file and return its path.
    """

    dataset = [
        {
            "url": f"{tmp_path}/images/cat.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.25990500000000005,
                    "ymin": 0.275424,
                    "xmax": 0.469097,
                    "ymax": 0.54661,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.0.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.5062115,
                    "ymin": 0.06405,
                    "xmax": 0.7655285000000001,
                    "ymax": 0.392562,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.1.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.15727049999999998,
                    "ymin": 0.11605750000000004,
                    "xmax": 0.7293355,
                    "ymax": 0.7737005,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.1.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.17749600000000001,
                    "ymin": 0.11215,
                    "xmax": 0.66878,
                    "ymax": 0.42783000000000004,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.2.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.149773,
                    "ymin": 0.0560185,
                    "xmax": 0.7456130000000001,
                    "ymax": 0.5226855,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.2.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.48969850000000004,
                    "ymin": 0.22767849999999998,
                    "xmax": 0.8462755,
                    "ymax": 0.5491075,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.3.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.029673000000000005,
                    "ymin": 0.04121849999999999,
                    "xmax": 0.424717,
                    "ymax": 0.4884015,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.3.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.060559500000000016,
                    "ymin": 0.04545450000000001,
                    "xmax": 0.4968945,
                    "ymax": 0.4586775,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.4.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.1879345,
                    "ymin": 0.07664750000000001,
                    "xmax": 0.4576135,
                    "ymax": 0.4390505,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.4.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.12422349999999999,
                    "ymin": 0.012987000000000026,
                    "xmax": 0.8524845000000001,
                    "ymax": 0.7483770000000001,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.5.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.3397625,
                    "ymin": 0.3765765,
                    "xmax": 0.7071095000000001,
                    "ymax": 0.8233855,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.5.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.245341,
                    "ymin": -5.000000000143778e-07,
                    "xmax": 0.580745,
                    "ymax": 0.3876285,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.6.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.15844250000000001,
                    "ymin": 0.395235,
                    "xmax": 0.47039549999999997,
                    "ymax": 0.766389,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.6.jpg",
            "subset": "TRAIN",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.07725850000000001,
                    "ymin": 0.1442465,
                    "xmax": 0.6351035,
                    "ymax": 0.8589955,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.7.jpg",
            "subset": "VALID",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.5252225,
                    "ymin": 0.036818500000000004,
                    "xmax": 0.9747775,
                    "ymax": 0.5080995,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.7.jpg",
            "subset": "VALID",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.25310599999999994,
                    "ymin": 0.1441125,
                    "xmax": 0.492236,
                    "ymax": 0.38664350000000003,
                },
                {
                    "label": "cat",
                    "xmin": 0.8509315000000001,
                    "ymin": 0.2478025,
                    "xmax": 0.9891305,
                    "ymax": 0.41827749999999997,
                },
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.8.jpg",
            "subset": "VALID",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.215133,
                    "ymin": 0.2480155,
                    "xmax": 0.504451,
                    "ymax": 0.6051584999999999,
                },
                {
                    "label": "cat",
                    "xmin": 0.4928545,
                    "ymin": 0.17437,
                    "xmax": 0.7130375,
                    "ymax": 0.50035,
                },
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.8.jpg",
            "subset": "VALID",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.20126749999999993,
                    "ymin": 0.14115900000000003,
                    "xmax": 0.9572105,
                    "ymax": 0.854383,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/cat.9.jpg",
            "subset": "TEST",
            "object_detection": [
                {
                    "label": "cat",
                    "xmin": 0.145235,
                    "ymin": 0.006833499999999992,
                    "xmax": 0.789713,
                    "ymax": 0.3280185,
                }
            ],
        },
        {
            "url": f"{tmp_path}/images/dog.9.jpg",
            "subset": "TEST",
            "object_detection": [
                {
                    "label": "dog",
                    "xmin": 0.328051,
                    "ymin": 0.06293699999999999,
                    "xmax": 0.965135,
                    "ymax": 0.510489,
                }
            ],
        },
    ]
    p = tmp_path / "MOCK_DATASET_OD.json"
    p.write_text(json.dumps(dataset))
    create_image_files(dataset)
    return str(p)
