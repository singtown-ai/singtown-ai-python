from .type import Annotation, TaskResponse
from pathlib import Path

dataset_dir = Path(__file__).parent.joinpath("dataset").absolute()

MOCK_DATASET_CF = [
    {
        "file": dataset_dir.joinpath("cat.0.jpg").as_posix(),
        "type": "train",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.1.jpg").as_posix(),
        "type": "train",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.2.jpg").as_posix(),
        "type": "train",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.3.jpg").as_posix(),
        "type": "train",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.4.jpg").as_posix(),
        "type": "train",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.5.jpg").as_posix(),
        "type": "train",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.6.jpg").as_posix(),
        "type": "train",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.7.jpg").as_posix(),
        "type": "val",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.8.jpg").as_posix(),
        "type": "val",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("cat.9.jpg").as_posix(),
        "type": "test",
        "classification": "cat",
    },
    {
        "file": dataset_dir.joinpath("dog.0.jpg").as_posix(),
        "type": "train",
        "classification": "dog",
    },
    {
        "file": dataset_dir.joinpath("dog.1.jpg").as_posix(),
        "type": "train",
        "classification": "dog",
    },
    {
        "file": dataset_dir.joinpath("dog.2.jpg").as_posix(),
        "type": "train",
        "classification": "dog",
    },
    {
        "file": dataset_dir.joinpath("dog.3.jpg").as_posix(),
        "type": "train",
        "classification": "dog",
    },
    {
        "file": dataset_dir.joinpath("dog.4.jpg").as_posix(),
        "type": "train",
        "classification": "dog",
    },
    {
        "file": dataset_dir.joinpath("dog.5.jpg").as_posix(),
        "type": "val",
        "classification": "dog",
    },
    {
        "file": dataset_dir.joinpath("dog.6.jpg").as_posix(),
        "type": "val",
        "classification": "dog",
    },
    {
        "file": dataset_dir.joinpath("dog.7.jpg").as_posix(),
        "type": "test",
        "classification": "dog",
    },
]

MOCK_DATASET_OD = [
    {
        "file": dataset_dir.joinpath("cat.0.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.1.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.2.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.3.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.4.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.5.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.6.jpg").as_posix(),
        "type": "val",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.7.jpg").as_posix(),
        "type": "val",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.8.jpg").as_posix(),
        "type": "val",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("cat.9.jpg").as_posix(),
        "type": "test",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.0.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.1.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.2.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.3.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.4.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.5.jpg").as_posix(),
        "type": "train",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.6.jpg").as_posix(),
        "type": "val",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.7.jpg").as_posix(),
        "type": "val",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.8.jpg").as_posix(),
        "type": "val",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "file": dataset_dir.joinpath("dog.9.jpg").as_posix(),
        "type": "test",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
]

MOCK_TASK_MAP = {
    "0": TaskResponse(
        project={
            "labels": ["cat", "dog"],
            "type": "CLASSIFICATION",
        },
        type="TRAIN",
        status="PENDING",
        params={
            "model": "MobileNetV2",
            "weight": "imagenet",
            "alpha": 0.35,
            "imgw": 96,
            "imgh": 96,
            "epochs": 1,
            "learning_rate": 0.001,
        },
    ),
    "1": TaskResponse(
        project={
            "labels": ["cat", "dog"],
            "type": "OBJECT_DETECTION",
        },
        type="TRAIN",
        status="PENDING",
        params={
            "model": "Yolov5s",
            "weight": "coco2017",
            "imgw": 640,
            "imgh": 640,
            "epochs": 1,
            "learning_rate": 0.001,
        },
    ),
    "2": TaskResponse(
        project={
            "labels": ["cat", "dog"],
            "type": "CLASSIFICATION",
        },
        type="DEPLOY",
        status="PENDING",
        trained_file="https://example.com/trained_model.zip",
        params={
            "imgw": 96,
            "imgh": 96,
        },
    ),
}

MOCK_DATASET_MAP = {
    "0": [Annotation(**item) for item in MOCK_DATASET_CF],
    "1": [Annotation(**item) for item in MOCK_DATASET_OD],
    "2": [Annotation(**item) for item in MOCK_DATASET_CF],
}
