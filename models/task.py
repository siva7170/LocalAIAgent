
from pydantic import BaseModel, Field
from typing import Any


class Task(BaseModel):

    id: int

    tool: str

    purpose: str

    arguments: dict[str, Any] = Field(default_factory=dict)

    requires_approval: bool = False
