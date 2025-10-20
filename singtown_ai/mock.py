from .type import Annotation, TaskResponse

MOCK_DATASET_CF = [
    {
        "url": "https://ai.singtown.com/media/cat.0.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.1.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.2.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.3.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.4.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.5.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.6.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.6.jpg",
        "subset": "TRAIN",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.7.jpg",
        "subset": "VALID",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.8.jpg",
        "subset": "VALID",
        "classification": "cat",
    },
    {
        "url": "https://ai.singtown.com/media/cat.9.jpg",
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
        "subset": "TRAIN",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.2.jpg",
        "subset": "TRAIN",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.3.jpg",
        "subset": "TRAIN",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.4.jpg",
        "subset": "TRAIN",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.5.jpg",
        "subset": "TRAIN",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.6.jpg",
        "subset": "TRAIN",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.7.jpg",
        "subset": "VALID",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.8.jpg",
        "subset": "VALID",
        "classification": "dog",
    },
    {
        "url": "https://ai.singtown.com/media/dog.9.jpg",
        "subset": "TEST",
        "classification": "dog",
    },
]

MOCK_DATASET_OD = [
    {
        "url": "https://ai.singtown.com/media/cat.0.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.1.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.2.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.3.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.4.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.4.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.6.jpg",
        "subset": "VALID",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.7.jpg",
        "subset": "VALID",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.8.jpg",
        "subset": "VALID",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/cat.9.jpg",
        "subset": "TEST",
        "object_detection": [
            {"label": "cat", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.0.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.1.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.2.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.3.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.4.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.5.jpg",
        "subset": "TRAIN",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.6.jpg",
        "subset": "VALID",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.7.jpg",
        "subset": "VALID",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.8.jpg",
        "subset": "VALID",
        "object_detection": [
            {"label": "dog", "bbox": [0, 0, 0.3, 0.4]},
        ],
    },
    {
        "url": "https://ai.singtown.com/media/dog.9.jpg",
        "subset": "TEST",
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
        trained_file="https://ai.singtown.com/media/trained_model.zip",
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
