from pydantic import BaseModel
from typing import List, Literal


TaskStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED"]
ProjectType = Literal["CLASSIFICATION", "OBJECT_DETECTION"]
TaskType = Literal["TRAIN", "DEPLOY"]
DatasetSubset = Literal["TRAIN", "VALID", "TEST"]


class ObjectDetectionEntry(BaseModel):
    label: str
    bbox: List[float]


class Annotation(BaseModel):
    url: str
    subset: DatasetSubset
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


class MockData(BaseModel):
    task: TaskResponse
    dataset: List[Annotation]
