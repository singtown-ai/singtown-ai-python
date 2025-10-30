from pydantic import BaseModel
from typing import List, Literal


class BoundingBox(BaseModel):
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Annotation(BaseModel):
    url: str
    subset: Literal["TRAIN", "VALID", "TEST"]
    classification: str = ""
    object_detection: List[BoundingBox] = []


class Project(BaseModel):
    labels: List[str]
    type: Literal["CLASSIFICATION", "OBJECT_DETECTION"]


class TaskResponse(BaseModel):
    project: Project
    status: Literal["PENDING", "RUNNING", "SUCCESS", "FAILED"] = "PENDING"
    device: str
    model_name: str
    freeze_backbone: bool
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping: int
    export_width: int
    export_height: int
    metrics: List[dict] = []


class LogEntry(BaseModel):
    timestamp: float
    content: str


class MockData(BaseModel):
    task: TaskResponse
    dataset: List[Annotation]
