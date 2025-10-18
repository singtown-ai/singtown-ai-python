from pydantic import BaseModel
from typing import List
from enum import Enum


class TaskStatus(str, Enum):
    pending = "PENDING"
    running = "RUNNING"
    success = "SUCCESS"
    failed = "FAILED"


class ProjectType(str, Enum):
    classification = "CLASSIFICATION"
    object_detection = "OBJECT_DETECTION"


class TaskType(str, Enum):
    train = "TRAIN"
    deploy = "DEPLOY"


class DatasetType(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class ObjectDetectionEntry(BaseModel):
    label: str
    bbox: List[float]


class Annotation(BaseModel):
    file: str
    type: DatasetType
    classification: str = ""
    object_detection: List[ObjectDetectionEntry] = []


class Project(BaseModel):
    labels: List[str]
    type: ProjectType


class TaskResponse(BaseModel):
    project: Project
    type: TaskType
    status: TaskStatus
    params: dict
    trained_file: str = ""
    metrics: List[dict] = []


class LogEntry(BaseModel):
    timestamp: float
    content: str
