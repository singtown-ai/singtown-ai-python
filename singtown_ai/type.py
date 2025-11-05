from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import List, Literal


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class BoundingBox(CamelCaseModel):
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Annotation(CamelCaseModel):
    url: str
    subset: Literal["TRAIN", "VALID", "TEST"]
    classification: str = ""
    object_detection: List[BoundingBox] = []


class Project(CamelCaseModel):
    labels: List[str]
    type: Literal["CLASSIFICATION", "OBJECT_DETECTION"]


class LogEntry(CamelCaseModel):
    timestamp: float
    content: str


class TaskResponse(CamelCaseModel):
    project: Project
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
    logs: List[LogEntry] = []
