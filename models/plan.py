
from pydantic import BaseModel, Field
from models.task import Task


class ExecutionPlan(BaseModel):

    understanding: str

    done: bool = False

    tasks: list[Task] = Field(default_factory=list)
